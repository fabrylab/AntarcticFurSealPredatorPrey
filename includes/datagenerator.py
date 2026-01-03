import warnings
warnings.filterwarnings("ignore")
import logging
import tifffile
logging.disable(logging.WARNING)

import tensorflow as tf
import numpy as np
import clickpoints
import os
import cv2

import clickpoints
from .images import imageLoader, tiffLoader

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class Datagenerator(tf.keras.utils.Sequence):
    def __init__(self, cdbPaths,
                 markers=("upright", "prone"),
                 w=512,
                 h=512,
                 anchorOverlap=2,
                 downsamples=(32, 8, 2),
                 batch_size=8,
                 keys="yolo",
                 outputValues=("o", "x", "y", "w", "h", "c"),
                 positive_oversample=0.,

                 flipX=False,
                 photoNoise=0.,
                 photoOffset=0.,
                 photoGamma=0.,
                 photoMinMax=(0, 255),
                 scaling=1,
                 noAugmentation=False,
                 ):

        self.w = int(w)
        self.h = int(h)
        self.outputValues = outputValues
        self.anchorOverlap = anchorOverlap

        # self.paths = paths
        self.positive_oversample = positive_oversample
        self.cdbPaths = cdbPaths

        self.downsamples = np.array(downsamples)
        self.downsamplesN = np.array([[self.w // d, self.h // d] for d in self.downsamples], dtype=int)

        self.batch_size = int(batch_size)
        self.markers = markers

        if isinstance(keys, str):
            self.keys = [keys + str(i) for i in range(len(self.downsamples))]
        elif isinstance(keys, (list, tuple)):
            assert len(keys) == len(self.downsamples), "Lengths of downsample scales and key list do not match!"
            self.keys = keys
        else:
            raise ValueError("Could not interpret keys!")

        self.noAugmentation = noAugmentation
        self.flipX = flipX
        self.photoNoise = photoNoise
        self.photoOffset = photoOffset
        self.photoGamma = photoGamma
        self.photoMinMax = photoMinMax
        self.scaling = np.array(scaling)

        self.X = np.zeros((self.batch_size, self.h, self.w, 1, 3))

        self.databases = {}
        self.data = {}
        self.metaData = {}
        self.images = {}
        self.imagePaths = {}
        self.imageShape = {}
        self.imageLoaders = {}
        self.cams = {}

        i = 0
        for f in cdbPaths:
            cdb = clickpoints.DataFile(f)

            imagesIds = set([m.image.sort_index for m in cdb.getRectangles(type=self.markers)])

            for imageId in imagesIds:
                i += 1
                #print(i)
                self.databases[i] = cdb
                cdbImg = cdb.getImage(frame=imageId)
                self.imagePaths[i] = os.path.abspath(
                    os.path.join(os.path.split(f)[0], cdbImg.path.path, cdbImg.filename))
                assert os.path.isfile(self.imagePaths[i]), "Could not find image path! %s" % self.imagePaths[i]
                if self.imagePaths[i].endswith("tiff"):
                    self.imageLoaders[i] = tiffLoader(self.imagePaths[i])
                else:
                    self.imageLoaders[i] = imageLoader(self.imagePaths[i])

                self.imageShape[i] = (
                self.imageLoaders[i].h_input, self.imageLoaders[i].w_input, self.imageLoaders[i].depth_input)

                CDBmarkers = {}
                for j, markerName in enumerate(self.markers):
                    type_id = cdb.getMarkerType(name=markerName).id
                    boxes = np.array(cdb.db.execute_sql(
                        "SELECT x,y,width,height from rectangle where type_id=%s" % type_id).fetchall())
                    if len(boxes) < 1:
                        CDBmarkers[markerName] = np.zeros((0, 5))
                        continue
                    rectHeights = boxes[:, 3]
                    rectWidths = boxes[:, 2]
                    positions = boxes[:, :2]+boxes[:,2:]/2

                    CDBmarkers[markerName] = np.concatenate([positions,  #
                                                             rectWidths[:, None],
                                                             rectHeights[:, None],
                                                             self.markers.index(markerName) * np.ones_like(
                                                                 rectHeights[:, None]),
                                                             # anchorLevel[:,None],
                                                             ], axis=1)
                self.data[i] = CDBmarkers
            cdb.db.close()

        self.nOutputs = 0
        if "o" in self.outputValues:
            self.nOutputs += 1
        if "x" in self.outputValues:
            self.nOutputs += 1
        if "y" in self.outputValues:
            self.nOutputs += 1
        if "w" in self.outputValues:
            self.nOutputs += 1
        if "h" in self.outputValues:
            self.nOutputs += 1
        if "c" in self.outputValues:
            self.nOutputs += len(self.markers)

    def getOutputIndices(self, outputValues, markers):
        i = 0
        outputIndizes = {}
        for k in outputValues:
            if k in ["c","n"]:
                outputIndizes[k] = slice(i, i+len(markers))
                i +=len(markers)
            else:
                outputIndizes[k] = slice(i, i+1)
                i +=1
        return outputIndizes
    def getOutputLength(self, outputValues, markers):
        nOutputs = 0
        if "o" in outputValues:
            nOutputs += 1
        if "x" in outputValues:
            nOutputs += 1
        if "y" in outputValues:
            nOutputs += 1
        if "w" in outputValues:
            nOutputs += 1
        if "h" in outputValues:
            nOutputs += 1
        if "c" in outputValues:
            nOutputs += len(markers)
        if "n" in outputValues:
            nOutputs += len(markers)
        return nOutputs

    def getMarkerLength(self, outputValues, nOutputs):
        if not any(["c" in outputValues, "n" in outputValues]):
            raise ValueError("Can not detemine number of markers without class or number output.")
        nO = int(nOutputs)
        if "o" in outputValues:
            nO -= 1
        if "x" in outputValues:
            nO -= 1
        if "y" in outputValues:
            nO -= 1
        if "w" in outputValues:
            nO -= 1
        if "h" in outputValues:
            nO -= 1
        divisor = sum(["c" in outputValues, "n" in outputValues])
        mLength = int(nO//divisor)
        assert nOutputs==self.getOutputLength(outputValues=outputValues, markers=list(range(mLength))), "Could not reconstruct markerlength"
        return mLength

    def __len__(self):
        #return len(positions)//self.batch_size//100
        return max(len(self.data)//self.batch_size, 1)

    def __getSample__(self, image_id=None, clicker=None, pos_X=None, pos_Y=None, force_positive=False, scaling=1):
        if image_id is None:
            assert all([k is None for k in [clicker, pos_X, pos_Y]])
            image_id = np.random.choice(list(self.data.keys()))
        data = np.concatenate([self.data[image_id][m] for m in self.markers], axis=0)
        s = scaling
        ws = int(self.w/scaling)
        hs = int(self.h/scaling)
        positions, rectHeights, rectWidths, types = np.split(data, (2, 3, 4), axis=1)
        anchorLevelIds = np.sum((rectHeights*s < (np.array(self.downsamples)[None,:-1] / self.anchorOverlap)), axis=1)

        H, W, _ = self.imageShape[image_id]
        if force_positive and (pos_X is None or pos_Y is None):
            levelID = np.random.choice(np.unique(anchorLevelIds))
            allowedPos, = np.where(anchorLevelIds==levelID)
            rId = allowedPos[np.random.randint(len(allowedPos))]
            pos_X, pos_Y = positions[rId]
            pos_X = max(0, min(pos_X - ws // 2, W - ws))
            pos_Y = max(0, min(pos_Y - hs // 2, H - hs))
        else:
            if pos_X is None:
                pos_X = np.random.randint(W - ws)
            if pos_Y is None:
                pos_Y = np.random.randint(H - hs)

        slicePositionMask = np.all(positions > np.array([pos_X, pos_Y]), axis=1) & np.all(
            positions < np.array([pos_X + ws, pos_Y + hs]), axis=1)

        anchorLevelIds = anchorLevelIds[slicePositionMask]
        anchorSizes = self.downsamples[anchorLevelIds.astype(int)]

        anchorPosId = (((positions[slicePositionMask] - np.array([[pos_X, pos_Y]])) * s)// anchorSizes[:,None]).astype(int)
        anchorPos = (((positions[slicePositionMask] - np.array([[pos_X, pos_Y]])) * s) % anchorSizes[:,None]) / anchorSizes[:,None]
        anchorW = rectHeights[slicePositionMask] * s / anchorSizes[:,None]
        anchorH = rectWidths[slicePositionMask] * s / anchorSizes[:,None]
        anchorT = types[slicePositionMask] == np.arange(len(self.markers))[None, :]

        grid = dict(
            [[self.keys[j], np.zeros((1, ASX, ASY, self.nOutputs), dtype=np.float32)] for j, (ASX, ASY) in
             enumerate(self.downsamplesN)])
        for j, levelId, (pIdX, pIdY), (x, y), w, h, t in zip(range(len(anchorLevelIds)), anchorLevelIds, anchorPosId,
                                                          anchorPos, anchorW, anchorH, anchorT):
            if "o" in self.outputValues:
                grid[self.keys[int(levelId)]][0, pIdY, pIdX, self.outputIndizes["o"]] = 1
            if "x" in self.outputValues:
                grid[self.keys[int(levelId)]][0, pIdY, pIdX, self.outputIndizes["x"]] = x
            if "y" in self.outputValues:
                grid[self.keys[int(levelId)]][0, pIdY, pIdX, self.outputIndizes["y"]] = y
            if "w" in self.outputValues:
                grid[self.keys[int(levelId)]][0, pIdY, pIdX, self.outputIndizes["w"]] = w
            if "h" in self.outputValues:
                grid[self.keys[int(levelId)]][0, pIdY, pIdX, self.outputIndizes["h"]] = h
            if "c" in self.outputValues:
                grid[self.keys[int(levelId)]][0, pIdY, pIdX, self.outputIndizes["c"]] = t
            if "n" in self.outputValues:
                grid[self.keys[int(levelId)]][0, pIdY, pIdX, self.outputIndizes["n"]] += t

        imSlice, (px, py, ww, hh) = self.imageLoaders[image_id].getSlice(scaling=1., offX_orig=pos_X, offY_orig=pos_Y,
                                                                         # height_orig=self.h,
                                                                         # width_orig=self.w
                                                                         height_orig=hs,
                                                                         width_orig=ws,
                                                                         )
        if s!=1:
            if s>1:
                imSlice = cv2.resize(imSlice, (self.h, self.w), interpolation=cv2.INTER_LINEAR)
            else:
                imSlice = cv2.resize(imSlice, (self.h, self.w), interpolation=cv2.INTER_NEAREST)
        return imSlice, grid

    def __getitem__(self, index):
        bigGrid = dict(
            [[self.keys[j], np.zeros((self.batch_size, ASX, ASY, self.nOutputs), dtype=np.float32)] for j, (ASX, ASY) in
             enumerate(self.downsamplesN)])
        self.X[:] = 0

        for b in range(self.batch_size):

            if self.scaling!=1:
                if len(self.scaling.shape)==0 or len(self.scaling)<2:
                    scaleDir = np.random.randint(low=0, high=2, size=1) -1
                    scale = np.random.uniform(low=1, high=self.scaling)**scaleDir
                else:
                    scaleDir = (np.random.randint(low=-self.scaling[0], high=self.scaling[1], size=N) > 0) * 2 - 1
                    scale = np.random.uniform(low=1, high=self.scaling[int(scaleDir>0)])**scaleDir
            else:
                scale = 1
            if self.noAugmentation:
                scale=1

            if ((index==0) and (b==0)) or (np.random.rand()<self.positive_oversample):
                imSlice, grid = self.__getSample__(force_positive=True, scaling=scale)
            else:
                imSlice, grid = self.__getSample__(scaling=scale)
            
            
            if not self.noAugmentation:
                aug = False

                if self.flipX:
                    if np.random.rand()>=0.5:
                        imSlice = imSlice[:,::-1]

                if self.photoGamma>1:
                    aug = True
                    gammaDir = 2*np.random.randint(low=0,high=2,size=1)-1
                    gamma = np.random.uniform(low=1, high=self.photoGamma, size=1)**gammaDir
                    # print("----", imSlice.dtype)
                    gammaLUT = np.round((np.arange(256.)/255.)**gamma * 255.)
                    imSliceA = gammaLUT[imSlice]
                    imSliceB = (imSlice/255)**gamma * 255
                
                imSlice = imSlice.astype(self.X.dtype)

                if self.photoOffset>0:
                    aug = True
                    offset = np.random.uniform(low=0, high=self.photoOffset, size=1)
                    imSlice = (imSlice+offset)*(255/(255+offset))
                    offset = np.random.uniform(low=0, high=self.photoOffset, size=1)
                    imSlice = imSlice*(255/(255+offset))

                if self.photoNoise>0:
                    aug = True
                    imSlice = imSlice + np.random.normal(loc=0, scale=self.photoNoise, size=imSlice.shape).astype(np.int8)

                if aug:
                    imSlice = np.round(np.maximum(self.photoMinMax[0], np.minimum(imSlice, self.photoMinMax[1])))
            imSlice = imSlice.astype(self.X.dtype)

            try:
                self.X[b] = imSlice[:,:,None,:]
                for k in bigGrid:
                    bigGrid[k][b] = grid[k][0]
            except ValueError:
                self.X[b] = self.X[(b+1)%self.batch_size]
                for k in bigGrid:
                    bigGrid[k][b] = bigGrid[k][(b+1)%self.batch_size]

        return {"Image": self.X[:,:,:,0,:]}, dict([[k, v[:,:,:,:]] for k,v in bigGrid.items()])

    def genValSet(self, force_positive=True):
        out = []
        for image_id in self.data.keys():
            data = np.concatenate([self.data[image_id][m] for m in self.markers], axis=0)
            positions, rectHeights, rectWidths, types = np.split(data, (2, 3, 4), axis=1)
            anchorLevelIds = np.sum((rectHeights < (np.array(self.downsamples)[None, :-1] / self.anchorOverlap)),
                                    axis=1)
            H, W,_ = self.imageShape[image_id]
            if force_positive:
                rId = np.random.randint(len(positions))
                pos_X, pos_Y = positions[rId]
                pos_X = max(0, min(pos_X - self.w // 2, W - self.w))
                pos_Y = max(0, min(pos_Y - self.h // 2, H - self.h))
            else:
                pos_X = np.random.randint(W - self.w)
                pos_Y = np.random.randint(H - self.h)

            out.append({
                "image_id":image_id,
                "pos_X": pos_X,
                "pos_Y": pos_Y,
            })
        return out

    def getVal(self, valDataDict):
        bigGrid = dict(
            [[self.keys[j], np.zeros((len(valDataDict), ASX, ASY, self.nOutputs), dtype=np.float32)] for j, (ASX, ASY) in
             enumerate(self.downsamplesN)])
        X = np.zeros((len(valDataDict), *self.X.shape[1:]), dtype=self.X.dtype)
        for b,e in enumerate(valDataDict):
            if e["image_id"] not in self.data:
                print("Image with ID", e["image_id"], "does not exist! Skipping.")
                continue
            imSlice, grid = self.__getSample__(**e)
            X[b] = imSlice[:, :, None, :]
            for k in bigGrid:
                bigGrid[k][b] = grid[k][0]
        return {"Image": X[:,:,:,0,:]}, dict([[k, v[:,:,:,:]] for k,v in bigGrid.items()])

class HyrarchicalDatagenerator(Datagenerator):
    def __init__(self, *args, typeHierarchy={}, **kwargs):
        self.typeHierarchy = typeHierarchy
        levelLabels = {}
        labelParents = {}
        def unpackLevels(key, entry, level):
            if level not in levelLabels:
                levelLabels[level] = []
            for k,v in entry.items():
                if k not in labelParents:
                    labelParents[k] = []
                if key is not None:
                    labelParents[k].extend([key, *labelParents[key]])
                levelLabels[level].append(k)
                unpackLevels(k, v, level+1)
        unpackLevels(None, typeHierarchy, 0)
        vectorDict = {}
        levelLengths = [len(v) for v in levelLabels.values()]
        vectorLength = sum(levelLengths)
        i = 0
        levelVectorDict = {}
        for level, labels in levelLabels.items():
            levelVectorDict[level] = np.zeros(vectorLength, dtype=int)
            for l in labels:
                v = np.zeros(vectorLength, dtype=bool)
                v[i] = 1
                levelVectorDict[level][i] = 1
                i+=1
                for parent in labelParents[l]:
                    v |= vectorDict[parent].astype(bool)
                vectorDict[l] = v.astype(int)
        
        maxLevel = max(levelLabels)
        if len(levelLabels[maxLevel])<1:
            levelLabels.pop(maxLevel)
            levelVectorDict.pop(maxLevel)
    
        self.levelLabels = levelLabels
        self.labelParents = labelParents
        self.vectorDict = vectorDict
        self.levelVectorDict = levelVectorDict
        self.vectorLength = vectorLength
        
        super(HyrarchicalDatagenerator, self).__init__(*args, **kwargs)
        self.vectors = np.array([np.zeros(self.vectorLength), *[self.vectorDict[m] for m in self.markers]])
        self.outputIndizes = self.getOutputIndices(self.outputValues, self.markers)
        self.outputLength = self.getOutputLength(self.outputValues, self.markers)
        self.nOutputs_old = int(self.nOutputs)
        self.nOutputs = int(self.outputLength)
        
    def getOutputIndices(self, outputValues, markers):
        i = 0
        outputIndizes = {}
        for k in outputValues:
            if k in ["c","n"]:
                outputIndizes[k] = slice(i, i+self.vectorLength)
                i +=len(markers)
            else:
                outputIndizes[k] = slice(i, i+1)
                i +=1
        return outputIndizes
    def getOutputLength(self, outputValues, markers):
        nOutputs = 0
        if "o" in outputValues:
            nOutputs += 1
        if "x" in outputValues:
            nOutputs += 1
        if "y" in outputValues:
            nOutputs += 1
        if "w" in outputValues:
            nOutputs += 1
        if "h" in outputValues:
            nOutputs += 1
        if "c" in outputValues:
            nOutputs += self.vectorLength
        if "n" in outputValues:
            nOutputs += self.vectorLength
        return nOutputs
    def __getSample__(self, *args, **kwargs):
        self.nOutputs = int(self.nOutputs_old)
        imSlice, grid = super(HyrarchicalDatagenerator, self).__getSample__(*args, **kwargs)
        self.nOutputs = int(self.outputLength)
        #return imSlice, grid
        if "c" not in self.outputIndizes:
            return imSlice, grid
        new_grid = {}
        for key, v in grid.items():
            v2 = np.zeros(v.shape[:-1] + (self.outputLength,), dtype=v.dtype)
            for val, indices in self.outputIndizes.items():
                if val == "c":
                    continue
                v2[...,indices] = v[...,indices]
            c = v[...,self.outputIndizes["c"]]
            c_ = np.sum(np.cumsum(c[...,::-1], axis=-1), axis=-1).astype(int)
            c2 = np.zeros(c.shape[:-1] +(self.vectorLength,), dtype=c.dtype)
            c2[c_>0] = self.vectors[c_[c_>0]]
            v2[...,self.outputIndizes["c"]] = c2
            new_grid[key] = v2
        return imSlice, new_grid
