import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow_addons as tfa
try:
    from keras import backend as K
except:
    from tensorflow.keras import backend as K
from .darknet import darknet

import numpy as np
from datetime import datetime
import re
import sys
import os
    

MAXEXPFLOAT32=88.72283

def sigmoid(x):
    return 1/(1+np.exp(-np.asarray(x)))
np.sigmoid = sigmoid
np.clip_by_value = np.clip
tf.isinf = tf.math.is_inf
tf.isnan = tf.math.is_nan

class YoloDecoder():
    def __init__(self, outputValues=("o","x","y","w","h","c"),
                 projectTime=True, crop=0, isActivated=False, outputIndizes=None, nClasses=None, **kwargs):
        self.outputValues = outputValues
        self.projectTime = projectTime
        self.crop = crop
        self.isActivated = isActivated
        self.outputIndizes = outputIndizes
        self.nClasses = nClasses

    def decode(self, y_true, y_pred, returnNumpy=False, outputValues=None):
        decoded_true = self.decodeTrue(y_true, returnNumpy=returnNumpy, outputValues=outputValues)
        decoded_pred = self.decodePred(y_pred, returnNumpy=returnNumpy, outputValues=outputValues)
        return [(t,p) for t,p in zip(decoded_true, decoded_pred)]

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
        return mLength

    def getOutputIndizes(self, outputValues, markers):
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

    def decodePred(self, y_pred, returnNumpy=False, outputValues=None):
        if self.projectTime:
            y_pred = y_pred[:,:,:,0,:]
        if self.crop>0:
            assert self.crop<0.5, "Crop must be between 0 and 0.5"
            n, h, w, c= y_pred.shape
            cw = int(self.crop*w)
            ch = int(self.crop*h)
            y_pred = y_pred[:,ch:-ch,cw:-cw]

        oV = self.outputValues

        nFilters = y_pred.shape[-1]
        if self.nClasses is None:
            nClasses = self.getMarkerLength(outputValues=self.outputValues, nOutputs=nFilters)
        else:
            nClasses = self.nClasses
        if self.outputIndizes is None:
            outputIndizes = self.getOutputIndizes(outputValues=self.outputValues, markers=list(range(nClasses)))
        else:
            outputIndizes = self.outputIndizes

        out = {}
        if returnNumpy:
            y_pred = np.array(y_pred)
            backend = np
        else:
            backend = tf

        if self.isActivated:
            posx_pred = y_pred[..., outputIndizes["x"]] if "x" in oV else 0.5 * backend.ones_like(y_pred[..., 0])
        else:
            posx_pred = backend.sigmoid(y_pred[..., outputIndizes["x"]]) if "x" in oV else 0.5 * backend.ones_like(y_pred[..., 0])
        out["x"] = posx_pred

        if self.isActivated:
            posy_pred = y_pred[..., outputIndizes["y"]] if "y" in oV else 0.5 * backend.ones_like(y_pred[..., 0])
        else:
            posy_pred = backend.sigmoid(y_pred[..., outputIndizes["y"]]) if "y" in oV else 0.5 * backend.ones_like(y_pred[..., 0])
        out["y"] = posy_pred

        if self.isActivated:
            posw_pred = y_pred[..., outputIndizes["w"]] if "w" in oV else backend.ones_like(y_pred[..., 0])
        else:
            posw_pred = backend.exp(backend.minimum(MAXEXPFLOAT32, y_pred[..., outputIndizes["w"]])) if "w" in oV else backend.ones_like(y_pred[..., 0])
        out["w"] = backend.clip_by_value(posw_pred, 0., 1e4)

        if self.isActivated:
            posh_pred = y_pred[..., outputIndizes["h"]] if "h" in oV else backend.ones_like(y_pred[..., 0])
        else:
            posh_pred = backend.exp(backend.minimum(MAXEXPFLOAT32, y_pred[..., outputIndizes["h"]])) if "h" in oV else backend.ones_like(y_pred[..., 0])
        out["h"] = backend.clip_by_value(posh_pred, 0., 1e4)

        if self.isActivated:
            o_pred = y_pred[..., outputIndizes["o"]] if "o" in oV else backend.ones_like(y_pred[..., 0])
        else:
            o_pred = backend.sigmoid(y_pred[..., outputIndizes["o"]]) if "o" in oV else backend.ones_like(y_pred[..., 0])
        out["o"] = o_pred

        if self.isActivated:
            c_pred = y_pred[..., outputIndizes["c"]] if "c" in oV else backend.ones_like(y_pred[..., 0])
        else:
            c_pred = backend.sigmoid(y_pred[..., outputIndizes["c"]]) if "c" in oV else backend.ones_like(y_pred[..., 0])
        out["c"] = c_pred

        if self.isActivated:
            n_pred = y_pred[..., outputIndizes["n"]] if "n" in oV else backend.ones_like(y_pred[..., 0])
        else:
            # n_pred = backend.exp(backend.minimum(MAXEXPFLOAT32, y_pred[..., outputIndizes["n"]])) if "n" in oV else backend.zeros_like(y_pred[..., 0])
            n_pred = y_pred[..., outputIndizes["n"]] if "n" in oV else backend.ones_like(y_pred[..., 0])
        out["n"] = n_pred

        if outputValues is None:
            outputValues = self.outputValues
        return [out[o] for o in outputValues]
    
    def decodeTrue(self, y_true, returnNumpy=False, outputValues=None):
        if self.crop > 0:
            assert self.crop < 0.5, "Crop must be between 0 and 0.5"
            n, h, w, c = y_true.shape
            cw = int(self.crop * w)
            ch = int(self.crop * h)
            y_true = y_true[:,ch:-ch,cw:-cw]

        oV = self.outputValues
        if returnNumpy:
            y_true = np.asarray(y_true)
            backend = np
        else:
            backend = tf

        nFilters = y_true.shape[-1]
        if self.nClasses is None:
            nClasses = self.getMarkerLength(outputValues=self.outputValues, nOutputs=nFilters)
        else:
            nClasses = self.nClasses
        if self.outputIndizes is None:
            outputIndizes = self.getOutputIndizes(outputValues=self.outputValues, markers=list(range(nClasses)))
        else:
            outputIndizes = self.outputIndizes

        out = {}
        posx_true = y_true[..., outputIndizes["x"]] if "x" in oV else 0.5 * backend.ones_like(y_true[..., 0])
        out["x"] = posx_true

        posy_true = y_true[..., outputIndizes["y"]] if "y" in oV else 0.5 * backend.ones_like(y_true[..., 0])
        out["y"] = posy_true

        posw_true = y_true[..., outputIndizes["w"]] if "w" in oV else backend.ones_like(y_true[..., 0])
        out["w"] = posw_true

        posh_true = y_true[..., outputIndizes["h"]] if "h" in oV else backend.ones_like(y_true[..., 0])
        out["h"] = posh_true

        o_true = y_true[...,  outputIndizes["o"]] if "o" in oV else backend.ones_like(y_true[..., 0])
        out["o"] = o_true

        c_true = y_true[...,  outputIndizes["c"]] if "c" in oV else backend.ones_like(y_true[..., 0])
        out["c"] = c_true

        c_true = y_true[...,  outputIndizes["n"]] if "n" in oV else backend.ones_like(y_true[..., 0])
        out["n"] = c_true

        if outputValues is None:
            outputValues = self.outputValues
        return [out[o] for o in outputValues]
    
regexStr = "".join([
    r"^",
    r"(?P<timestamp>\d{8}-\d{6})",
    r"(_(?P<netName>[^_]*))",
    r"(_(?P<netType>[^_\W]*))?",
    r"(_(?P<yoloname>[^_\W]*))?",
    r"(_(?P<WH>\d*\.\d*-\d*\.\d))?",
    r"(_(?P<downsamples>\d{1,3}(-\d{1,3})*))",
    r"((_(?P<anchorOverlap>\d*(\.\d*)?))(-(?P<anchorChoice>[^_\W]*))?)?",
    r"(_(?P<mode>[^_\W]*))",
    r"(_(?P<markers>[^_\W]*(-[^_\W]*)*))?",
    r"(_(?P<outputValues>[oxywhcn]*))?",
    r"(_(?P<weights>[^_\W]*))",
    r"\.h5$"
    ])

filename_regex = re.compile(regexStr)

def parseWeightFilename(weightFilename):
    gd = filename_regex.match(os.path.split(weightFilename)[-1]).groupdict()
    kwargs = {}
    kwargs["ts"] = datetime.strptime(gd["timestamp"], "%Y%m%d-%H%M%S")
    kwargs["netName"] = gd["netName"]
    kwargs["netType"] = gd["netType"] if "netType" in gd else "simple"
    kwargs["yoloname"] = gd["yoloname"] if "yoloname" in gd else "yolo"
    W,H = [float(d) for d in gd["WH"].split("-")] if "WH" in gd else [0.8, 0.8]
    kwargs["w_p"]=W
    kwargs["h_p"]=H
    kwargs["downsamples"] = [int(d) for d in gd["downsamples"].split("-")]
    kwargs["anchorOverlap"] = float(gd["anchorOverlap"]) if "anchorOverlap" in gd else 1.
    kwargs["anchorChoice"] = gd["anchorChoice"] if "anchorChoice" in gd else "simple"
    kwargs["mode"] = str(gd["mode"])
    kwargs["markers"] = [str(d) for d in gd["markers"].split("-")] if "markers" in gd else ["Adult", "Chick"]
    kwargs["outputValues"] = list(gd["outputValues"])
    kwargs["weightsType"] = str(gd["weights"])
    if kwargs["anchorChoice"] == "density":
        kwargs["anchorOptimal"] = False
        kwargs["anchorFromDensity"] = True
    elif kwargs["anchorChoice"] == "optimal":
        kwargs["anchorOptimal"] = True
        kwargs["anchorFromDensity"] = False
    else:
        kwargs["anchorOptimal"] = False
        kwargs["anchorFromDensity"] = False
    kwargs["keys"] =  ["yolo"+str(d).zfill(3) for d in kwargs["downsamples"]]
    return kwargs

##

dataRegexStr = "".join([
    r"^",
    r"(?P<timestamp>\d{8}-\d{6})?",
    r"(?P<dataSetName>[^_]*)?",
    r"(_(?P<wh>\d*\.\d*-\d*\.\d))?",
    r"(_(?P<WH>\d*-\d*))?",
    r"(_(?P<downsamples>\d{1,3}(-\d{1,3})*))",
    r"((_(?P<anchorOverlap>\d*(\.\d*)?))(-(?P<anchorChoice>[^_\W]*))?)?",
    r"(_(?P<markers>[^_\W]*(-[^_\W]*)*))?",
    r"(_(?P<outputValues>[oxywhcn]*))?",
    r"(_(?P<scaling>\d*(\.\d*)?))?",
    r"(_(?P<positive_oversampling>\d*(\.\d*)?))?",
    r"\.np(z|y)$"
    ])
data_filename_regex = re.compile(dataRegexStr)

def parseDataSetFilename(data_path):
    dg_kwargs = {}
    gd = data_filename_regex.match(os.path.split(data_path)[-1]).groupdict()
    # dg_kwargs["timestamp"] = datetime.strptime(gd["timestamp"], "%Y%m%d-%H%M%S")
    dg_kwargs["h"] = float(gd["WH"].split("-")[0])
    dg_kwargs["w"] = float(gd["WH"].split("-")[1])
    dg_kwargs["w_p"] = float(gd["wh"].split("-")[0])
    dg_kwargs["h_p"] = float(gd["wh"].split("-")[1])
    dg_kwargs["downsamples"] = [int(ds) for ds in gd["downsamples"].split("-")]
    dg_kwargs["keys"] = ["yolo"+str(ds).zfill(3) for ds in dg_kwargs["downsamples"]]
    dg_kwargs["anchorOverlap"] = float(gd["anchorOverlap"])
    dg_kwargs["outputValues"] = list(gd["outputValues"])
    dg_kwargs["scaling"] = float(gd["scaling"]) if gd["scaling"] is not None else 1.
    dg_kwargs["positive_oversample"] = float(gd["positive_oversampling"]) if gd["positive_oversampling"] is not None else 1.
    if gd["anchorChoice"] == "density":
        dg_kwargs["anchorFromDensity"] = True
        dg_kwargs["anchorOptimal"] = False
        dg_kwargs["anchorOther"] = False
    elif gd["anchorChoice"] == "optimal":
        dg_kwargs["anchorFromDensity"] = False
        dg_kwargs["anchorOptimal"] = True
        dg_kwargs["anchorOther"] = False
    elif gd["anchorChoice"] == "other":
        dg_kwargs["anchorFromDensity"] = False
        dg_kwargs["anchorOptimal"] = False
        dg_kwargs["anchorOther"] = True
    else:
        dg_kwargs["anchorFromDensity"] = False
        dg_kwargs["anchorOptimal"] = False
        dg_kwargs["anchorOther"] = False
    return dg_kwargs
##
def getPretrainedFromName(netName, input_tensor):
    if netName == "mobilenet-1-00-224":
        net = tf.keras.applications.MobileNet(
            input_tensor=input_tensor,
            alpha=1.0,
            depth_multiplier=1,
            dropout=0.001,
            include_top=False,
            weights="imagenet",
            pooling=None,
        )
    elif netName == "mobilenetv2-1-00-224":
        net = tf.keras.applications.MobileNetV2(
            input_tensor=input_tensor,
            alpha=1.0,
            include_top=False,
            weights="imagenet",
            pooling=None,
        )
    elif netName == "MobilenetV3Small":
        net_pre = tf.keras.applications.MobileNetV3Small(
            input_tensor=input_tensor,
            alpha=1.0,
            include_top=False,
            weights="imagenet",
            pooling=None,
            include_preprocessing=False,
        )
        net = tf.keras.Model(name=netName, inputs=input_tensor, outputs=net_pre.get_layer(index=227).output)
    elif netName == "MobilenetV3Large":
        net_pre = tf.keras.applications.MobileNetV3Large(
            input_tensor=input_tensor,
            alpha=1.0,
            include_top=False,
            weights="imagenet",
            pooling=None,
            include_preprocessing=False,
        )
        net = tf.keras.Model(name=netName, inputs=input_tensor, outputs=net_pre.get_layer(index=261).output)
    elif netName.startswith("resnet"):
        kwargs = dict(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor,
            pooling=None, )
        if netName.endswith("50"):
            net = tf.keras.applications.ResNet50(**kwargs)
        elif netName.endswith("101"):
            net = tf.keras.applications.ResNet101(**kwargs)
        elif netName.endswith("152"):
            net = tf.keras.applications.ResNet152(**kwargs)
        elif netName.endswith("50v2"):
            net = tf.keras.applications.ResNet50V2(**kwargs)
        elif netName.endswith("101v2"):
            net = tf.keras.applications.ResNet101V2(**kwargs)
        elif netName.endswith("152v2"):
            net = tf.keras.applications.ResNet152V2(**kwargs)
        else:
            raise ValueError("Don't know net %s" % netName)
    elif netName.startswith("efficientnet"):
        kwargs = dict(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            pooling=None,
        )
        if netName.endswith("b0"):
            net = tf.keras.applications.efficientnet.EfficientNetB0(**kwargs)
        elif netName.endswith("b1"):
            net = tf.keras.applications.efficientnet.EfficientNetB1(**kwargs)
        elif netName.endswith("b2"):
            net = tf.keras.applications.efficientnet.EfficientNetB2(**kwargs)
        elif netName.endswith("b3"):
            net = tf.keras.applications.efficientnet.EfficientNetB3(**kwargs)
        elif netName.endswith("b4"):
            net = tf.keras.applications.efficientnet.EfficientNetB4(**kwargs)
        elif netName.endswith("b5"):
            net = tf.keras.applications.efficientnet.EfficientNetB5(**kwargs)
        elif netName.endswith("b6"):
            net = tf.keras.applications.efficientnet.EfficientNetB6(**kwargs)
        elif netName.endswith("b7"):
            net = tf.keras.applications.efficientnet.EfficientNetB7(**kwargs)
        else:
            raise ValueError("Don't know net %s" % netName)
    elif netName.startswith("densenet"):
        kwargs = dict(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            pooling=None,
        )
        if netName.endswith("121"):
            net = tf.keras.applications.DenseNet121(**kwargs)
        elif netName.endswith("169"):
            net = tf.keras.applications.DenseNet169(**kwargs)
        elif netName.endswith("201"):
            net = tf.keras.applications.DenseNet201(**kwargs)
        else:
            raise ValueError("Don't know net %s" % netName)
    elif netName == "inception-v3":
        net = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor,
            pooling=None,
        )
    elif netName == "inception-resnet-v2":
        net = tf.keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor,
            pooling=None,
        )
    # TODO: find a solution for big and small NAS net that have the same net.name
    # elif netName == "NASNet":
    #     net =tf.keras.applications.NASNetLarge(
    #         include_top=False,
    #         weights="imagenet",
    #         input_tensor=input_tensor,
    #         pooling=None,
    #     )
    elif netName == "NASNet":
        net = tf.keras.applications.NASNetMobile(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor,
            pooling=None,
        )
    elif netName == "xception":
        net = tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor,
            pooling=None,
        )
    elif netName == "vgg16":
        net = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor,
            pooling=None,
        )
    elif netName == "vgg19":
        net = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=input_tensor,
            pooling=None,
        )
    elif netName == "Darknet53":
        net = darknet(inputs=input_tensor)
        net.load_weights(os.path.join(os.path.split(__file__)[0], "darknet53_weights.h5"), by_name=True)
    else:
        raise ValueError("Don't know net %s" % netName)
    return net
##

layerDict = {
    "NASNet":{
        128: 768,
        64: 768,
        32: 768,
        16: 583,
        8: 346,
        4: 109,
        2: 48,
        1: 48,
    },
    "densenet121":{
        128:"relu",
        64: "relu",
        32: "relu",
        16: "pool4_conv",
        8: "pool3_conv",
        4: "pool2_conv",
        2: "conv1/relu",
        1: "conv1/relu",
    },
    "densenet169":{
        128:"relu",
        64: "relu",
        32: "relu",
        16: "pool4_conv",
        8: "pool3_conv",
        4: "pool2_conv",
        2: "conv1/relu",
        1: "conv1/relu",
    },
    "densenet201":{
        128:"relu",
        64: "relu",
        32: "relu",
        16: "pool4_conv",
        8: "pool3_conv",
        4: "pool2_conv",
        2: "conv1/relu",
        1: "conv1/relu",
    },
    "mobilenetv2_1.00_224":{
        128:"out_relu",
        64:"out_relu",
        32:"out_relu",
        16:"block_13_expand_relu",
        8:"block_6_expand_relu",
        4:"block_3_expand_relu",
        2:"block_1_expand_relu",
        1:"block_1_expand_relu",
    },
    "mobilenet_1.00_224":{
        128:"conv_pw_13_relu",
        64:"conv_pw_13_relu",
        32:"conv_pw_13_relu",
        16:"conv_pw_11_relu",
        8:"conv_pw_5_relu",
        4:"conv_pw_3_relu",
        2:"conv_pw_1_relu",
        1:"conv_pw_1_relu",
    },
    "MobilenetV3Large":{
        128:261,
        64:261,
        32:261,
        16:192,
        8:87,
        4:33,
        2:15,
        1:15,
    },
    "MobilenetV3Small":{
        128:227,
        64:227,
        32:227,
        16:158,
        8:44,
        4:23,
        2:6,
        1:6,
    },
    "efficientnetb0":{
        128:"top_activation",
        64:"top_activation",
        32:"top_activation",
        16:"block6a_expand_activation",
        8:"block4a_expand_activation",
        4:"block3a_expand_activation",
        2:"block2a_expand_activation",
        1:"block2a_expand_activation",
    },
    "efficientnetb1":{
        128:"top_activation",
        64:"top_activation",
        32:"top_activation",
        16:"block6a_expand_activation",
        8:"block4a_expand_activation",
        4:"block3a_expand_activation",
        2:"block2a_expand_activation",
        1:"block2a_expand_activation",
    },
    "efficientnetb2":{
        128:"top_activation",
        64:"top_activation",
        32:"top_activation",
        16:"block6a_expand_activation",
        8:"block4a_expand_activation",
        4:"block3a_expand_activation",
        2:"block2a_expand_activation",
        1:"block2a_expand_activation",
    },
    "efficientnetb3":{
        128:"top_activation",
        64:"top_activation",
        32:"top_activation",
        16:"block6a_expand_activation",
        8:"block4a_expand_activation",
        4:"block3a_expand_activation",
        2:"block2a_expand_activation",
        1:"block2a_expand_activation",
    },
    "efficientnetb4":{
        128:"top_activation",
        64:"top_activation",
        32:"top_activation",
        16:"block6a_expand_activation",
        8:"block4a_expand_activation",
        4:"block3a_expand_activation",
        2:"block2a_expand_activation",
        1:"block2a_expand_activation",
    },
    "efficientnetb5":{
        128:"top_activation",
        64:"top_activation",
        32:"top_activation",
        16:"block6a_expand_activation",
        8:"block4a_expand_activation",
        4:"block3a_expand_activation",
        2:"block2a_expand_activation",
        1:"block2a_expand_activation",
    },
    "efficientnetb6":{
        128:"top_activation",
        64:"top_activation",
        32:"top_activation",
        16:"block6a_expand_activation",
        8:"block4a_expand_activation",
        4:"block3a_expand_activation",
        2:"block2a_expand_activation",
        1:"block2a_expand_activation",
    },
    "efficientnetb7":{
        128:"top_activation",
        64:"top_activation",
        32:"top_activation",
        16:"block6a_expand_activation",
        8:"block4a_expand_activation",
        4:"block3a_expand_activation",
        2:"block2a_expand_activation",
        1:"block2a_expand_activation",
    },
    "Darknet53": {
        128: "act_52",
        64: "act_52",
        32: "act_52",
        16: "act_43",
        8: "act_26",
        4: "act_9",
        2: "act_4",
        1: "LRELU_0",
    }
}

def addYoloLayers(net, downsamples, nFilters, netType="simple", yoloName="yolo", activations=None,
                  kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None, returnBboxes=False,):
    if activations is None:
        activations = dict([[k, "linear"] for k in range(nFilters)])
    else:
        assert len(activations) == nFilters, "Not enough activations for all filters!"
    yolos = {}
    for ds in downsamples:
        # _,h,w,c = net.output_shape
        layerHint = layerDict[net.name][ds]
        if isinstance(layerHint, int):
            layer = net.get_layer(index=layerHint)
        else:
            layer = net.get_layer(name=layerHint)
        dsNet = net.input_shape[-2] / net.output_shape[-2]
        dsLayer = net.input_shape[-2] / layer.output_shape[-2]
        nFiltersLayer = 32  # layer.output_shape[-1]
        name = yoloName + str(ds).zfill(3)
        name2 = str(ds).zfill(3)
        if netType == "simple":
            if ds >= dsNet:
                ks = int(ds / dsNet)
                yolos[ds] = tf.keras.layers.Conv2D(name=name, kernel_size=(ks, ks),
                                                   strides=(ks, ks), filters=nFilters, activation="linear",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                   )(net.output)
            else:
                ks = int(dsNet / ds)
                yolos[ds] = tf.keras.layers.Conv2DTranspose(name=name, kernel_size=(ks, ks),
                                                            strides=(ks, ks), filters=nFilters, activation="linear",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(net.output)
        elif netType == "layer":
            if ds >= dsLayer:
                ks = int(ds / dsLayer)
                yolos[ds] = tf.keras.layers.Conv2D(name=name, kernel_size=(ks, ks),
                                                   strides=(ks, ks), filters=nFilters, activation="linear",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(
                    layer.output)
            else:
                ks = int(dsLayer / ds)
                yolos[ds] = tf.keras.layers.Conv2DTranspose(name=name, kernel_size=(ks, ks),
                                                            strides=(ks, ks), filters=nFilters, activation="linear",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(
                    layer.output)
            print(ds, ks)
        elif netType == "skipConv":
            if ds >= dsNet:
                ks = int(ds / dsNet)
                upsampled = tf.keras.layers.Conv2D(name="ups" + name2, kernel_size=(ks, ks),
                                                   strides=(ks, ks), filters=nFilters, activation="linear",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(net.output)
            else:
                ks = int(dsNet / ds)
                upsampled = tf.keras.layers.Conv2DTranspose(name="ups" + name2, kernel_size=(ks, ks),
                                                            strides=(ks, ks), filters=nFilters, activation="linear",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(
                    net.output)
            if ds >= dsLayer:
                ks = int(ds / dsLayer)
                fromLayer = tf.keras.layers.Conv2D(name="fromLayer" + name2, kernel_size=(ks, ks),
                                                   strides=(ks, ks), filters=nFilters, activation="linear",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(layer.output)
            else:
                ks = int(dsLayer / ds)
                fromLayer = tf.keras.layers.Conv2DTranspose(name="fromLayer" + name2, kernel_size=(ks, ks),
                                                            strides=(ks, ks), filters=nFilters, activation="linear",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(
                    layer.output)
            conc = tf.keras.layers.concatenate([upsampled, fromLayer])
            yoloFilters = []
            for i in range(nFilters):
                yoloFilters.append(
                    tf.keras.layers.Conv2D(name=name+"_filter%d"%i, kernel_size=(1, 1),
                                           strides=(1, 1), filters=1, activation=activations[i],
                                           kernel_regularizer=kernel_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           activity_regularizer=activity_regularizer,
                                          )(conc)
                )
            yolos[ds] = tf.keras.layers.concatenate(yoloFilters, axis=-1, name=name)

        elif netType == "skipUps":
            if ds >= dsLayer:
                ks = int(ds / dsLayer)
                fromLayer = tf.keras.layers.AveragePooling2D(name="fromLayer" + name2, pool_size=(ks, ks))(
                    layer.output)
            else:
                ks = int(dsLayer / ds)
                fromLayer = tf.keras.layers.UpSampling2D(name="fromLayer" + name2, size=(ks, ks))(
                    layer.output)
            if ds >= dsNet:
                ks = int(ds / dsNet)
                upsampled = tf.keras.layers.AveragePooling2D(name="ups" + name2, pool_size=(ks, ks))(
                    net.output)
            else:
                ks = int(dsNet / ds)
                upsampled = tf.keras.layers.UpSampling2D(name="ups" + name2, size=(ks, ks))(net.output)
            x = tf.keras.layers.Conv2D(kernel_size=(3, 3), strides=(1, 1), filters=nFilters, activation="relu",
                                       padding="same",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(upsampled)
            conc = tf.keras.layers.concatenate([x, fromLayer])
            yolos[ds] = tf.keras.layers.Conv2D(name=name, kernel_size=(1, 1), strides=(1, 1),
                                               filters=nFilters, activation="linear", padding="same",
                                                   kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                            )(conc)
    if returnBboxes:
        bboxes = []
        for ds in downsamples:
            NX,NY = net.input_shape[1]//ds , net.input_shape[2]//ds
            bboxes.append(tf.reshape(yolos[ds], (-1, NX*NY, nFilters)))
        yolos["bboxes"] = tf.concat(bboxes, axis=1)
    return yolos

def logit(x):
    return - tf.math.log(1. / x - 1.)

def addSoftMax(downsamples, yolos, softMaxName="yolo"):
    maxDS = max(downsamples)
    bigF = 0.
    bigN = 0.
    for ds in downsamples:
        bigF += tf.keras.layers.MaxPool2D(pool_size=(maxDS//ds,maxDS//ds), strides=(maxDS//ds,maxDS//ds))(
            tf.sigmoid(yolos[ds][...,:1])
        )
        bigN += 1
    yolos2 = {}
    smallF = 1/(bigF/bigN)
    for ds in downsamples:

        yolos2[ds] = tf.keras.layers.concatenate(name=softMaxName+str(ds).zfill(3),inputs=[
            -tf.math.log((1+tf.exp(-yolos[ds][...,:1]))*tf.keras.layers.UpSampling2D(size=(maxDS//ds,maxDS//ds))(smallF)-1),
            yolos[ds][..., 1:],
        ], axis=-1)

    return yolos2

def buildNetwork(weightFilename, sliceW = 512,sliceH = 512):
    basename = os.path.splitext(os.path.split(weightFilename)[-1])[0]
    outputValues = list(basename.split("_")[-2])
    outputClasses = basename.split("_")[-3].split("-")
    nOutputs = 0
    for v in outputValues:
        if v in ["c","n"]:
            nOutputs += len(outputClasses)
        else:
            nOutputs += 1

    downsamples = np.array(basename.split("_")[-6].split("-"), dtype=int)
    yoloName = basename.split("_")[-8]
    netType = basename.split("_")[-9]
    netName = basename.split("_")[1]

    input_tensor = tf.keras.Input((sliceH, sliceW, 3), name="Image", )

    net = getPretrainedFromName(netName=netName, input_tensor=input_tensor)

    yolos = addYoloLayers(net, downsamples, nOutputs, netType=netType, yoloName=yoloName)

    if yoloName == "preyolo":
        yolos2 = addSoftMax(downsamples, yolos)
        model = tf.keras.Model(inputs=net.input, outputs=yolos2.values())
    else:
        model = tf.keras.Model(inputs=net.input, outputs=yolos.values())
    while True:
        try:
            model.load_weights(weightFilename, by_name=True)
            break
        except BlockingIOError:
            continue
    return model

def buildNetworkAde(weightFilename, sliceW = 512,sliceH = 512, batch_size=None):
    basename = os.path.splitext(os.path.split(weightFilename)[-1])[0]
    outputValues = list(basename.split("_")[-2])
    outputClasses = basename.split("_")[-3].split("-")
    if "c" in outputValues:
        nOutputs = len(outputValues) + len(outputClasses) - 1
    else:
        nOutputs = len(outputValues)

    downsamples = np.array(basename.split("_")[-6].split("-"), dtype=int)
    yoloName = basename.split("_")[-7]
    netType = basename.split("_")[-8]
    netName = basename.split("_")[1]

    input_tensor = tf.keras.Input((sliceH, sliceW, 3), name="Image", batch_size=batch_size)

    net = getPretrainedFromName(netName=netName, input_tensor=input_tensor)

    yolos = addYoloLayers(net, downsamples, nOutputs, netType=netType, yoloName=yoloName)

    if yoloName == "preyolo":
        yolos2 = addSoftMax(downsamples, yolos)
        model = tf.keras.Model(inputs=net.input, outputs=yolos2.values())
    else:
        model = tf.keras.Model(inputs=net.input, outputs=yolos.values())
    while True:
        try:
            model.load_weights(weightFilename, by_name=True)
            break
        except BlockingIOError:
            continue
    return model

def buildNetworkAdeWithDecoder(weightFilename, threshold,
                               sliceW = 512,sliceH = 512, nImages = 2, H=2592, W=4608, overlapY=0., overlapX=0.):
    threshold = np.array(threshold, ndmin=1)
    basename = os.path.splitext(os.path.split(weightFilename)[-1])[0]
    downsamples = np.array(basename.split("_")[-6].split("-"), dtype=int)
    ii, yy, xx = np.meshgrid(
        np.arange(nImages),
        np.arange(0, H - (sliceH - overlapY) + 1, (sliceH - overlapY), dtype=int),
        np.arange(0, W - (sliceW - overlapX) + 1, (sliceW - overlapX), dtype=int)
    )
    yy = np.minimum(yy.flatten(), H - sliceH)
    xx = np.minimum(xx.flatten(), W - sliceW)
    ii = np.minimum(ii.flatten(), nImages)
    batchsize = len(ii)

    model = buildNetworkAde(weightFilename=weightFilename, sliceH=sliceH, sliceW=sliceW, batch_size=batchsize)

    thresholdTensor = tf.convert_to_tensor(threshold, dtype=tf.float32)
    xxTensor = tf.convert_to_tensor(xx, dtype=tf.float32)
    yyTensor = tf.convert_to_tensor(yy, dtype=tf.float32)
    iiTensor = tf.convert_to_tensor(ii, dtype=tf.float32)
    downsamplesTensor = tf.convert_to_tensor(downsamples, dtype=tf.float32)

    x = []
    y = []
    w = []
    h = []
    c = []
    imgId = []
    for i, p in enumerate(model.outputs):
        oP, xP, yP, wP, hP, cP = tf.split(p, (1, 1, 1, 1, 1, 2), axis=-1)
        oP = tf.sigmoid(oP)
        xP = tf.sigmoid(xP)
        yP = tf.sigmoid(yP)
        cP = tf.sigmoid(cP)
        wP = tf.exp(wP)
        hP = tf.exp(hP)

        if len(threshold) > 1:
            m = (oP > thresholdTensor[i])
        else:
            m = (oP > thresholdTensor)

        b_off, y_off, x_off, _ = tf.transpose(tf.where(m))
        x_off = tf.cast(x_off, dtype=tf.float32)
        y_off = tf.cast(y_off, dtype=tf.float32)

        xx_ = tf.where(m, xxTensor[:, None, None, None], -1)[m]
        yy_ = tf.where(m, yyTensor[:, None, None, None], -1)[m]
        ii_ = tf.where(m, iiTensor[:, None, None, None], -1)[m]

        cx_ = ((xP[m] + x_off) * downsamplesTensor[i] + xx_)
        cy_ = ((yP[m] + y_off) * downsamplesTensor[i] + yy_)
        w_ = (wP[m] * downsamplesTensor[i])
        h_ = (hP[m] * downsamplesTensor[i])
        c_ = tf.argmax(cP, axis=-1)[m[:, :, :, 0]]

        x_ = cx_ #- w_ / 2
        y_ = cy_ #- h_ / 2

        x.append(x_)
        y.append(y_)
        w.append(w_)
        h.append(h_)
        c.append(c_)
        imgId.append(ii_)
    x = tf.concat(x, axis=0)
    y = tf.concat(y, axis=0)
    w = tf.concat(w, axis=0)
    h = tf.concat(h, axis=0)
    c = tf.concat(c, axis=0)
    imgId = tf.concat(imgId, axis=0)

    modelFull = tf.keras.Model(inputs=model.inputs, outputs=[x, y, w, h, c, imgId])
    return modelFull

def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float32':
        number_size = 4.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
##
def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
def get_max_memory():
    local_device_protos = device_lib.list_local_devices()
    return dict([[x.name, x.memory_limit] for x in local_device_protos if x.device_type == 'GPU'])
