import imageio
import cv2
import tifffile
import numpy as np
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
import os
from packaging import version

import warnings
warnings.filterwarnings("ignore")

NEWTIFFFILEVERSION = version.parse(tifffile.__version__) >= version.parse("2020.9.22")

def getImageShape(path):
    """
    Method to access image shape faster than loading the whole image.
    :param path: String containing the file path.
    :return: tuple of (height, width, depth)
    """
    shape = None
    PIL_formats = ["jpg", "jpeg", "png"]
    ffmpeg_formats = ["mp4"]
    if any([path.lower().endswith(f) for f in PIL_formats]):
        with open(path, "rb") as f:
            ImPar = ImageFile.Parser()
            chunk = f.read(2048)
            count = 2048
            while chunk != "":
                ImPar.feed(chunk)
                if ImPar.image:
                    break
                chunk = f.read(2048)
                count += 2048
            shape = ImPar.image.size[::-1] + (len(ImPar.image.getbands()),)
    elif path.lower().endswith("tif") or path.lower().endswith("tiff"):
        imPointer = tifffile.TiffFile(path)
        shape = imPointer.pages[0].shape
    elif any([path.lower().endswith(f) for f in ffmpeg_formats]):
        reader = cv2.VideoCapture(path)
        # TODO: get the proper image color space????? https://github.com/FFmpeg/FFmpeg/blob/master/libavutil/pixfmt.h
        return (reader.get(cv2.CAP_PROP_FRAME_HEIGHT), reader.get(cv2.CAP_PROP_FRAME_WIDTH), 3)
    else:
        raise ValueError("Format not known!")
    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)
    return shape

def checkPyramid(path):
    # load image and perform scaling
    tif = tifffile.TiffFile(path)
    pyramid = True
    level_dimensions = []
    for i, page in enumerate(tif.pages):
        # if we are not the first page
        if len(level_dimensions) and page.is_reduced is None:
            pyramid = False
        # append the dimensions
        level_dimensions.append((page.shape[1], page.shape[0]))
    if np.all(np.array(level_dimensions) == level_dimensions[0]) or (len(level_dimensions) == 1):
        pyramid = False
    return pyramid
##
def loadTiffSlice(path, scaling, offX_orig, offY_orig, height_orig, width_orig, fileObj=None):
    # load original image dimensions
    h_input, w_input, depth_input = getImageShape(path)
    # calculate crop offset in rescaled image
    offX_array = int(offX_orig * scaling)
    offY_array = int(offY_orig * scaling)
    # check for the boundaries of the crop offset in the rescaled image
    offX_array = max(0, min(offX_array, np.round(scaling*w_input)))
    offY_array = max(0, min(offY_array, np.round(scaling*h_input)))
    # calculate crop widht and height in scaled image
    outputarray_width = int(width_orig * scaling)
    outputarray_height = int(height_orig * scaling)
    # check for boundaries (do not overshoot image boundaries with crop width and height)
    outputarray_width = min(int(w_input*scaling), outputarray_width+offX_array) - offX_array
    outputarray_height = min(int(h_input*scaling), outputarray_height+offY_array) - offY_array
    #print(offX_array, offY_array, outputarray_width, outputarray_height)

    # load image and perform scaling
    if fileObj is not None:
        tif = tifffile.TiffFile(fileObj, name=path, offset=0)
    else:
        tif = tifffile.TiffFile(path)
    pyramid = True
    level_dimensions = []
    for i, page in enumerate(tif.pages):
        # if we are not the first page
        if len(level_dimensions) and page.is_reduced is None:
            pyramid = False
        # append the dimensions
        level_dimensions.append((page.shape[1], page.shape[0]))
    if np.all(np.array(level_dimensions) == level_dimensions[0]) or (len(level_dimensions) == 1):
        pyramid = False
        Warning("Not a pyramid tiff!", path)

    if not pyramid:
        im = tifffile.imread(path)
        if im.dtype.itemsize > 1:
            q1 = np.percentile(im, 1)
            q99 = np.percentile(im, 99)
            im = ((im - q1) * (255 / (q99 - q1)))
            im[im < 0] = 0
            im[im > 255] = 255
        if scaling != 1:
            im = cv2.resize(im, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)
        # perform the crop
        im = im[offY_array:offY_array + outputarray_height, offX_array:offX_array + outputarray_width]
    else:
        # store the level 0 dimension
        dimensions = level_dimensions[0]
        # calculate how much the pyramid pages downsample
        level_downsamples = np.array([dimensions[0] / dim[0] for dim in level_dimensions])
        # some logic to find the correct downsample layer
        # make a sort LUT for the downsample layers
        sortArgs = np.argsort(level_downsamples)[::-1]
        # now find the downsample level, that matches our scaling (the one that oversamples our target sampling the least)
        targetLevelId = sortArgs[np.argmin(np.maximum(scaling*level_downsamples[sortArgs],1))]
        page = tif.pages[targetLevelId]
        # we get a new scaling factor, that covers teh difference between the requested scale and the scale of our downsample
        scalingFactor = scaling*level_downsamples[targetLevelId]
        pageDimension = level_dimensions[targetLevelId]
        if not page.is_tiled:
            # if image is not tiled, we load the whole image
            im = page.asarray()
            # perform leftover resize
            im = cv2.resize(im, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_NEAREST)
            # perform the crop
            im = im[offY_array:offY_array + outputarray_height, offX_array:offX_array + outputarray_width]
        else:
            # scale our target crop to the actual scaling
            y1, y2, x1, x2 = (np.array([offY_array, offY_array + outputarray_height, offX_array, offX_array + outputarray_width])/scalingFactor).astype(int)
            # get the positions of the tiles in the file
            offsets, bytecounts = page.dataoffsets, page.databytecounts
            # initialize some lists
            slide_indices_shapes = []
            used_offsets = []
            used_bytecounts = []
            slide_rects = []

            # iterate over all tiles
            for i in range(len(offsets)):
                # decode with empty content to obtain the position and shape of the tile
                if NEWTIFFFILEVERSION:
                    segment, (_, _, iy, ix, _), (_, wy, wx, _) = page.decode(None, i)
                else:
                    segment, (_, _, _, iy, ix, _), (_, wy, wx, _) = page.decode(None, i)
                # check if it overlaps with the target region
                if any([(ix + wx)<x1, ix>x2, (iy+wy)<y1, iy>y2]):
                    continue
                # store the offsets
                used_offsets.append(offsets[i])
                used_bytecounts.append(bytecounts[i])

                slide_indices_shapes.append(i)
                # slide_rects.append([indices[3], indices[4], indices[3] + shape[1], indices[4] + shape[2]])
                slide_rects.append([ix, iy, ix+wx, iy+wy])

            # determine the region covered by the tiles
            slide_rects = np.array(slide_rects)
            sx1, sy1 = np.min(slide_rects[:, :2], axis=0)
            sx2, sy2 = np.max(slide_rects[:, 2:], axis=0)
            # new crop positions, within the tile boundaries
            cx1 = x1-sx1
            cx2 = x2-sx1
            cy1 = y1-sy1
            cy2 = y2-sy1

            # and initialize an array accordingly
            im = np.zeros((sy2 - sy1, sx2 - sx1, 1 if len(page.shape) == 2 else page.shape[2]), dtype=page.dtype)
            #print(im.shape, "from", len(slide_rects), "page",pageDimension, "level", targetLevelId)

            # decode the tiles
            decodeargs = {}
            if 347 in page.keyframe.tags:
                # decodeargs["tables"] = page._gettags({347}, lock=None)[0][1].value
                decodeargs["jpegtables"] = page._gettags({347}, lock=None)[0][1].value
            for seg, i in zip(
                    page.parent.filehandle.read_segments(used_offsets, used_bytecounts),
                    slide_indices_shapes):
                if NEWTIFFFILEVERSION:
                    segment, (_, _, iy, ix, _), (_, wy, wx, _) = page.decode(seg[0], i, **decodeargs)
                else:
                    segment, (_, _, _, iy, ix, _), (_, wy, wx, _) = page.decode(seg[0], i, **decodeargs)
                if not segment is None:
                    im[
                    iy-sy1:iy-sy1+wy,
                    ix-sx1:ix-sx1+wx
                    ] = segment
            # crop to target shape
            im = im[cy1:cy2,cx1:cx2]
            # optionally drop the channel dimension
            if len(page.shape) == 2:
                im = im[:, :, 0]
            if im.dtype.itemsize > 1:
                q1 = np.percentile(im, 1)
                q99 = np.percentile(im, 99)
                im = ((im - q1) * (255 / (q99 - q1)))
                im[im < 0] = 0
                im[im > 255] = 255
            # perform leftover resize
            if scalingFactor!=1:
                im = cv2.resize(im, None, fx=min(1, scalingFactor), fy=min(1, scalingFactor), interpolation=cv2.INTER_NEAREST)
            # perform the crop
            # im = im[offY_array:offY_array + outputarray_height, offX_array:offX_array + outputarray_width]
    im = im.astype(np.uint8)
    # return the image and the verified crop coordinates.
    return im, np.array([offX_array, offY_array, outputarray_width, outputarray_height])/scaling
##
class imageLoader():
    def __init__(self, path, buffer=True):
        self.path = path
        self.buffer = buffer
        if self.buffer:
            self.image = imageio.imread(self.path)
            self.h_input, self.w_input, self.depth_input = self.image.shape
        else:
            self.image=None
            self.h_input, self.w_input, self.depth_input= getImageShape(self.path)

    def getSlice(self, scaling, offX_orig, offY_orig, height_orig, width_orig):
        offX_orig = int(offX_orig)
        offY_orig = int(offY_orig)
        height_orig = int(height_orig)
        width_orig = int(width_orig)
        if self.buffer:
            return self.image[offY_orig:offY_orig+height_orig, offX_orig:offX_orig+width_orig], (offX_orig, offY_orig, height_orig, width_orig)
        else:
            image = imageio.imread(self.path)
            return image[offY_orig:offY_orig+height_orig, offX_orig:offX_orig+width_orig], (offX_orig, offY_orig, height_orig, width_orig)

##
import shutil
class tiffLoader():
    def __init__(self, path, bufferpath=None):
        self.path = path
        self.bufferpath = bufferpath
        if self.bufferpath is not None:
            os.makedirs(self.bufferpath, exist_ok=True)
            self.bufferfile = os.path.join(self.bufferpath, os.path.splitext(os.path.split(self.path)[-1])[0])
            if os.path.isdir(self.bufferfile):
                shutil.rmtree(self.bufferfile)
            if os.path.isfile(self.bufferfile):
                os.remove(self.bufferfile)
            os.makedirs(self.bufferfile, exist_ok=True)

        # load image and perform scaling
        # with open(path, "rb") as fileObj:
        #     tif = tifffile.TiffFile(fileObj, name=path, offset=0)
        tif = tifffile.TiffFile(self.path, offset=0)
        # load original image dimensions
        self.h_input, self.w_input, self.depth_input = tif.pages[0].shape
        self.dtype = tif.pages[0].dtype
        self.pyramid = True
        self.level_dimensions = []
        self.slide_indices_shapes = {}
        self.used_offsets = {}
        self.used_bytecounts = {}
        self.slide_rects = {}
        self.decodeargs = {}
        self.pagesDtypes = {}
        self.pagesShapes = {}
        for i, page in enumerate(tif.pages):
            # if we are not the first page
            if len(self.level_dimensions) and page.is_reduced is None:
                self.pyramid = False
            # append the dimensions
            self.level_dimensions.append((page.shape[1], page.shape[0]))
        if np.all(np.array(self.level_dimensions) == self.level_dimensions[0]) or (len(self.level_dimensions) == 1):
            self.pyramid = False
            Warning("Not a pyramid tiff!", path)
        if self.pyramid:
            # store the level 0 dimension
            dimensions = self.level_dimensions[0]
            # calculate how much the pyramid pages downsample
            level_downsamples = np.array([dimensions[0] / dim[0] for dim in self.level_dimensions])
            # some logic to find the correct downsample layer
            for targetLevelId in range(len(level_downsamples)):
                page = tif.pages[targetLevelId]
                if not page.is_tiled:
                    self.pyramid = False
                    Warning("Not a pyramid tiff!", path)
                else:
                    self.pagesShapes[targetLevelId] = page.shape
                    self.pagesDtypes[targetLevelId] = page.dtype
                    # get the positions of the tiles in the file
                    offsets, bytecounts = page.dataoffsets, page.databytecounts
                    # initialize some lists
                    self.slide_indices_shapes[targetLevelId] = []
                    self.used_offsets[targetLevelId] = []
                    self.used_bytecounts[targetLevelId] = []
                    self.slide_rects[targetLevelId] = []
                    self.decodeargs[targetLevelId] = {}

                    self.slide_indices_shapes[targetLevelId] = np.zeros((len(offsets),1), dtype=int)
                    self.used_offsets[targetLevelId] = np.zeros((len(offsets),1), dtype=int)
                    self.used_bytecounts[targetLevelId] = np.zeros((len(offsets),1), dtype=int)
                    self.slide_rects[targetLevelId] = np.zeros((len(offsets),4), dtype=int)
                    self.decodeargs[targetLevelId] = {}

                    # iterate over all tiles
                    for i in range(len(offsets)):
                        # decode with empty content to obtain the position and shape of the tile
                        if NEWTIFFFILEVERSION:
                            segment, (_, _, iy, ix, _), (_, wy, wx, _) = page.decode(None, i)
                        else:
                            segment, (_, _, _, iy, ix, _), (_, wy, wx, _) = page.decode(None, i)
                        # store the offsets
                        self.used_offsets[targetLevelId][i] = offsets[i]
                        self.used_bytecounts[targetLevelId][i] = bytecounts[i]
                        self.slide_indices_shapes[targetLevelId][i] = i
                        self.slide_rects[targetLevelId][i] = [ix, iy, wx, wy]

                    # decode the tiles
                    if 347 in page.keyframe.tags:
                        self.decodeargs[targetLevelId]["jpegtables"] = page._gettags({347}, lock=None)[0][1].value
        tif.close()
        self.buffered = []

    def __hashSLR__(self, slr):
        return "%016x%04x%04x%04x%04x"%(self.path.__hash__(), *slr)

    def getSlice(self, scaling, offX_orig, offY_orig, height_orig, width_orig):
        # load original image dimensions
        h_input, w_input, depth_input = self.h_input, self.w_input, self.depth_input
        # calculate crop offset in rescaled image
        offX_array = int(offX_orig * scaling)
        offY_array = int(offY_orig * scaling)
        # check for the boundaries of the crop offset in the rescaled image
        offX_array = max(0, min(offX_array, np.round(scaling * w_input)))
        offY_array = max(0, min(offY_array, np.round(scaling * h_input)))
        # calculate crop widht and height in scaled image
        outputarray_width = int(width_orig * scaling)
        outputarray_height = int(height_orig * scaling)
        # check for boundaries (do not overshoot image boundaries with crop width and height)
        outputarray_width = min(int(w_input * scaling), outputarray_width + offX_array) - offX_array
        outputarray_height = min(int(h_input * scaling), outputarray_height + offY_array) - offY_array
        # print(offX_array, offY_array, outputarray_width, outputarray_height)

        # load image and perform scaling
        if not self.pyramid:
            raise NotImplementedError("Not a pyramid tiff!: " + self.path)
        else:
            # store the level 0 dimension
            dimensions = self.level_dimensions[0]
            # calculate how much the pyramid pages downsample
            level_downsamples = np.array([dimensions[0] / dim[0] for dim in self.level_dimensions])
            # some logic to find the correct downsample layer
            # make a sort LUT for the downsample layers
            sortArgs = np.argsort(level_downsamples)[::-1]
            # now find the downsample level, that matches our scaling (the one that oversamples our target sampling the least)
            targetLevelId = sortArgs[np.argmin(np.maximum(scaling * level_downsamples[sortArgs], 1))]
            # we get a new scaling factor, that covers teh difference between the requested scale and the scale of our downsample
            scalingFactor = scaling * level_downsamples[targetLevelId]
            pageDimension = self.level_dimensions[targetLevelId]

            # scale our target crop to the actual scaling
            y1, y2, x1, x2 = (np.array([offY_array, offY_array + outputarray_height, offX_array, offX_array + outputarray_width])/scalingFactor).astype(int)

            ix,iy,wx,wy = self.slide_rects[targetLevelId].T
            slr_mask = ~np.any([(ix + wx) < x1, ix > x2, (iy + wy) < y1, iy > y2], axis=0)
            # slr_mask = ~(((slr[:,0]+slr[:,2])<x1)|(slr[:,0]>x2)|((slr[:,1]+slr[:,3])<y1)|(slr[:,1]>y2))

            slide_indices_shapes = np.asarray(self.slide_indices_shapes[targetLevelId])[slr_mask]
            used_offsets = np.asarray(self.used_offsets[targetLevelId])[slr_mask]
            used_bytecounts = np.asarray(self.used_bytecounts[targetLevelId])[slr_mask]

            slide_rects = np.asarray(self.slide_rects[targetLevelId])[slr_mask]
            # go from x1,y1,w,h to x1,y1,x2,y2
            slide_rects[:,2:] += slide_rects[:,:2]

            # determine the region covered by the tiles
            sx1, sy1 = np.min(slide_rects[:, :2], axis=0)
            sx2, sy2 = np.max(slide_rects[:, 2:], axis=0)
            # new crop positions, within the tile boundaries
            cx1 = x1-sx1
            cx2 = x2-sx1
            cy1 = y1-sy1
            cy2 = y2-sy1

            # initialize target array
            im = np.zeros((sy2 - sy1, sx2 - sx1, self.depth_input), dtype=self.dtype)
            bufferMask = np.array([offset in self.buffered for offset in used_offsets])
            if self.bufferpath is not None:
                for slr, offset in zip(slide_rects[bufferMask], used_offsets[bufferMask].flatten()):
                    ix, iy, ix1, iy1 = slr
                    wx = ix1-ix
                    wy = iy1-iy
                    # hash = self.buffered[tuple(slr)]
                    hash = "%016x"%(offset)
                    with open(os.path.join(self.bufferfile, hash), "rb") as mf:
                        # d = np.frombuffer(mf.read(), dtype=self.dtype).reshape((wy,wx,self.depth_input))
                        im[
                        iy - sy1:iy - sy1 + wy,
                        ix - sx1:ix - sx1 + wx
                        ] = np.frombuffer(mf.read(), dtype=self.dtype).reshape((wy,wx,self.depth_input))
                        # print("used buffer!")

            if np.any(~bufferMask):
                with open(self.path, "rb") as fileObj:
                    tif = tifffile.TiffFile(fileObj, name=self.path, offset=0)
                    page = tif.pages[targetLevelId]
                    # decode the tiles
                    decodeargs = self.decodeargs[targetLevelId]
                    for seg, i, offset in zip(
                            page.parent.filehandle.read_segments(list(used_offsets[~bufferMask].flatten()), list(used_bytecounts[~bufferMask].flatten())),
                            list(slide_indices_shapes[~bufferMask].flatten()),
                            list(used_offsets[~bufferMask].flatten()),
                    ):
                        if NEWTIFFFILEVERSION:
                            segment, (_, _, iy, ix, _), (_, wy, wx, _) = page.decode(seg[0], i, **decodeargs)
                        else:
                            segment, (_, _, _, iy, ix, _), (_, wy, wx, _) = page.decode(seg[0], i, **decodeargs)
                        if self.bufferpath is not None:
                            hash = "%016x" % (int(offset))
                            with open(os.path.join(self.bufferfile, hash), "wb") as mf:
                                mf.write(segment.tobytes())
                            self.buffered.append(offset)
                        im[
                        iy-sy1:iy-sy1+wy,
                        ix-sx1:ix-sx1+wx
                        ] = segment
            # crop to target shape
            im = im[cy1:cy2,cx1:cx2]
            # # optionally drop the channel dimension
            # if len(page.shape) == 2:
            #     im = im[:, :, 0]
            if im.dtype.itemsize > 1:
                q1 = np.percentile(im, 1)
                q99 = np.percentile(im, 99)
                im = ((im - q1) * (255 / (q99 - q1)))
                im[im < 0] = 0
                im[im > 255] = 255
            # perform leftover resize
            if scalingFactor!=1:
                im = cv2.resize(im, None, fx=min(1, scalingFactor), fy=min(1, scalingFactor), interpolation=cv2.INTER_NEAREST)
        im = im.astype(np.uint8)
        # return the image and the verified crop coordinates.
        return im, np.array([offX_array, offY_array, outputarray_width, outputarray_height])/scaling

def loadTiffSlice(path, scaling, offX_orig, offY_orig, height_orig, width_orig, fileObj=None):
    # load original image dimensions
    h_input, w_input, depth_input = getImageShape(path)
    # calculate crop offset in rescaled image
    offX_array = int(offX_orig * scaling)
    offY_array = int(offY_orig * scaling)
    # check for the boundaries of the crop offset in the rescaled image
    offX_array = max(0, min(offX_array, np.round(scaling*w_input)))
    offY_array = max(0, min(offY_array, np.round(scaling*h_input)))
    # calculate crop widht and height in scaled image
    outputarray_width = int(width_orig * scaling)
    outputarray_height = int(height_orig * scaling)
    # check for boundaries (do not overshoot image boundaries with crop width and height)
    outputarray_width = min(int(w_input*scaling), outputarray_width+offX_array) - offX_array
    outputarray_height = min(int(h_input*scaling), outputarray_height+offY_array) - offY_array
    #print(offX_array, offY_array, outputarray_width, outputarray_height)

    # load image and perform scaling
    if fileObj is not None:
        tif = tifffile.TiffFile(fileObj, name=path, offset=0)
    else:
        tif = tifffile.TiffFile(path)
    pyramid = True
    level_dimensions = []
    for i, page in enumerate(tif.pages):
        # if we are not the first page
        if len(level_dimensions) and page.is_reduced is None:
            pyramid = False
        # append the dimensions
        level_dimensions.append((page.shape[1], page.shape[0]))
    if np.all(np.array(level_dimensions) == level_dimensions[0]) or (len(level_dimensions) == 1):
        pyramid = False
        Warning("Not a pyramid tiff!", path)

    if not pyramid:
        im = tifffile.imread(path)
        if im.dtype.itemsize > 1:
            q1 = np.percentile(im, 1)
            q99 = np.percentile(im, 99)
            im = ((im - q1) * (255 / (q99 - q1)))
            im[im < 0] = 0
            im[im > 255] = 255
        if scaling != 1:
            im = cv2.resize(im, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)
        # perform the crop
        im = im[offY_array:offY_array + outputarray_height, offX_array:offX_array + outputarray_width]
    else:
        # store the level 0 dimension
        dimensions = level_dimensions[0]
        # calculate how much the pyramid pages downsample
        level_downsamples = np.array([dimensions[0] / dim[0] for dim in level_dimensions])
        # some logic to find the correct downsample layer
        # make a sort LUT for the downsample layers
        sortArgs = np.argsort(level_downsamples)[::-1]
        # now find the downsample level, that matches our scaling (the one that oversamples our target sampling the least)
        targetLevelId = sortArgs[np.argmin(np.maximum(scaling*level_downsamples[sortArgs],1))]
        page = tif.pages[targetLevelId]
        # we get a new scaling factor, that covers teh difference between the requested scale and the scale of our downsample
        scalingFactor = scaling*level_downsamples[targetLevelId]
        pageDimension = level_dimensions[targetLevelId]
        if not page.is_tiled:
            # if image is not tiled, we load the whole image
            im = page.asarray()
            # perform leftover resize
            im = cv2.resize(im, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_NEAREST)
            # perform the crop
            im = im[offY_array:offY_array + outputarray_height, offX_array:offX_array + outputarray_width]
        else:
            # scale our target crop to the actual scaling
            y1, y2, x1, x2 = (np.array([offY_array, offY_array + outputarray_height, offX_array, offX_array + outputarray_width])/scalingFactor).astype(int)
            # get the positions of the tiles in the file
            offsets, bytecounts = page.dataoffsets, page.databytecounts
            # initialize some lists
            slide_indices_shapes = []
            used_offsets = []
            used_bytecounts = []
            slide_rects = []

            # iterate over all tiles
            for i in range(len(offsets)):
                # decode with empty content to obtain the position and shape of the tile
                if NEWTIFFFILEVERSION:
                    segment, (_, _, iy, ix, _), (_, wy, wx, _) = page.decode(None, i)
                else:
                    segment, (_, _, _, iy, ix, _), (_, wy, wx, _) = page.decode(None, i)
                # check if it overlaps with the target region
                if any([(ix + wx)<x1, ix>x2, (iy+wy)<y1, iy>y2]):
                    continue
                # store the offsets
                used_offsets.append(offsets[i])
                used_bytecounts.append(bytecounts[i])

                slide_indices_shapes.append(i)
                # slide_rects.append([indices[3], indices[4], indices[3] + shape[1], indices[4] + shape[2]])
                slide_rects.append([ix, iy, ix+wx, iy+wy])

            # determine the region covered by the tiles
            slide_rects = np.array(slide_rects)
            sx1, sy1 = np.min(slide_rects[:, :2], axis=0)
            sx2, sy2 = np.max(slide_rects[:, 2:], axis=0)
            # new crop positions, within the tile boundaries
            cx1 = x1-sx1
            cx2 = x2-sx1
            cy1 = y1-sy1
            cy2 = y2-sy1

            # and initialize an array accordingly
            im = np.zeros((sy2 - sy1, sx2 - sx1, 1 if len(page.shape) == 2 else page.shape[2]), dtype=page.dtype)
            #print(im.shape, "from", len(slide_rects), "page",pageDimension, "level", targetLevelId)

            # decode the tiles
            decodeargs = {}
            if 347 in page.keyframe.tags:
                # decodeargs["tables"] = page._gettags({347}, lock=None)[0][1].value
                decodeargs["jpegtables"] = page._gettags({347}, lock=None)[0][1].value
            for seg, i in zip(
                    page.parent.filehandle.read_segments(used_offsets, used_bytecounts),
                    slide_indices_shapes):
                if NEWTIFFFILEVERSION:
                    segment, (_, _, iy, ix, _), (_, wy, wx, _) = page.decode(seg[0], i, **decodeargs)
                else:
                    segment, (_, _, _, iy, ix, _), (_, wy, wx, _) = page.decode(seg[0], i, **decodeargs)
                im[
                iy-sy1:iy-sy1+wy,
                ix-sx1:ix-sx1+wx
                ] = segment
            # crop to target shape
            im = im[cy1:cy2,cx1:cx2]
            # optionally drop the channel dimension
            if len(page.shape) == 2:
                im = im[:, :, 0]
            if im.dtype.itemsize > 1:
                q1 = np.percentile(im, 1)
                q99 = np.percentile(im, 99)
                im = ((im - q1) * (255 / (q99 - q1)))
                im[im < 0] = 0
                im[im > 255] = 255
            # perform leftover resize
            if scalingFactor!=1:
                im = cv2.resize(im, None, fx=min(1, scalingFactor), fy=min(1, scalingFactor), interpolation=cv2.INTER_NEAREST)
            # perform the crop
            # im = im[offY_array:offY_array + outputarray_height, offX_array:offX_array + outputarray_width]
    im = im.astype(np.uint8)
    # return the image and the verified crop coordinates.
    return im, np.array([offX_array, offY_array, outputarray_width, outputarray_height])/scaling