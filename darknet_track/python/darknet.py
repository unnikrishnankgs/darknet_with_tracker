from ctypes import *
import json

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

# Same as tAnnInfo
class ANNINFO(Structure):
    _fields_ = [
                ("x", c_int), 
                ("y", c_int), 
                ("w", c_int), 
                ("h", c_int), 
                ("pcClassName", c_char_p), 
                ("fCurrentFrameTimeStamp", c_double), 
                ("nVideoId", c_int),
                ("prob", c_double)
               ]

# Same as tfnRaiseAnnCb
RAISEANNFUNC = CFUNCTYPE(c_int, ANNINFO)

#Same as tDetectorModel
class DETECTORMODEL(Structure):
    _fields_ = [("pcCfg", c_char_p),
                ("pcWeights", c_char_p),
                ("pcFileName", c_char_p),
                ("pcDataCfg", c_char_p),
                ("fTargetFps", c_double),
                ("fThresh", c_double),
                ("pfnRaiseAnnCb", (RAISEANNFUNC)),
                ("nVideoId", c_int),
                ("isVideo", c_int)
               ]

#lib = CDLL("/Users/gotham/work/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/unnikrishnan/work/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

def load_meta(f):
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA
    return lib.get_metadata(f)

def load_net(cfg, weights):
    load_network = lib.load_network_p
    load_network.argtypes = [c_char_p, c_char_p, c_int]
    load_network.restype = c_void_p
    return load_network(cfg, weights, 0)

def load_img(f):
    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE
    return load_image(f, 0, 0)

def letterbox_img(im, w, h):
    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE
    return letterbox_image(im, w, h)

def predict(net, im):
    pred = lib.network_predict_image
    pred.argtypes = [c_void_p, IMAGE]
    pred.restype = POINTER(c_float)
    return pred(net, im)

def classify(net, meta, im):
    out = predict(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im):
    out = predict(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def cb(a):
    print("helloooo" + a)
    return 0

#CMPFUNC = CFUNCTYPE(c_int, POINTER(c_int), POINTER(c_int))
CMPFUNC = CFUNCTYPE(c_int, POINTER(c_int))

annotation_raised = {}
objects_raised = []

#[{"type": "Car", "keyframes": [{"continueInterpolation": true, "y": 501, "x": 420, "frame": 0, "h": 130, "w": 250}], "color": "#f28a9d", "user_info": {"user_id": "2"}}, {"type": "Car", "keyframes": [{"continueInterpolation": true, "y": 459, "x": 778, "frame": 0, "h": 93, "w": 177}], "color": "#efa58d", "user_info": {"user_id": "2"}}]

def raiseAnn(annInfo):
    x = int(annInfo.x)
    y = int(annInfo.y)
    w = int(annInfo.w)
    h = int(annInfo.h)
    fCurrentFrameTimeStamp = annInfo.fCurrentFrameTimeStamp
    
    print("got annInfo x:" + str(x) + " y:" + str(y) + " w:" + str(w) + " h:" + str(h) 
        + " fCurrentFrameTimeStamp:" + str(fCurrentFrameTimeStamp)
        )
    an_object = {}
    an_object['type'] = (annInfo.pcClassName).decode('utf-8')
    an_object['keyframes'] = []
    box = {}
    box['continueInterpolation'] = False;
    box['x'] = annInfo.x
    box['y'] = annInfo.y
    box['w'] = annInfo.w
    box['h'] = annInfo.h
    box['frame'] = annInfo.fCurrentFrameTimeStamp / 1000
    an_object['keyframes'].append(box)
    an_object['color'] = "#f28a9d"
    user_info = {}
    user_info['user_id'] = "2"
    an_object['user_info'] = user_info
    objects_raised.append(an_object)
    return 0

def test():
    run_detector_model = lib.run_detector_model
    run_detector_model.argtypes = [POINTER(DETECTORMODEL)]
    run_detector_model.restype = c_int;
    detectorModel = DETECTORMODEL("/home/unnikrishnan/work/darknet/cfg/yolo.cfg".encode('utf-8'), "/home/unnikrishnan/work/darknet/yolo.weights".encode('utf-8'), 
        #"/home/unnikrishnan/work/va/annotator/static/res/videos/1080p_WALSH_ST_000.mp4".encode('utf-8'),
        "/home/unnikrishnan/work/va/annotator/static/res/image_list/1080p_WALSH_ST_0602_000/1080p_WALSH_ST_0602_000_00001.jpeg".encode('utf-8'),
        #"/Users/gotham/work/research/video_annotation_and_deeplearning/BeaverDam/annotator/static/res/videos/trimmed_walsh_night.mp4".encode('utf-8'),
        #"/Users/gotham/work//research/videos_demo/trimmed_walsh_night.mp4".encode('utf-8'),
        #"/Users/gotham/work/research/video_annotation_and_deeplearning/videos/test_darknet.mp4".encode('utf-8'),
        "/home/unnikrishnan/work/darknet/cfg/coco.data".encode('utf-8'), 
        1, 
        0.24, 
        RAISEANNFUNC(raiseAnn), 0, 
        1);
    #print("pcDataCfg is "  + detectorModel.pcDataCfg.decode("utf-8"));
    return run_detector_model(pointer(detectorModel))

if __name__ == "__main__":
    test()
    print("JSON: " + json.dumps(objects_raised));
    #net = load_net("../cfg/yolo.cfg", "../yolo.weights")
    #im = load_img("data/wolf.jpg")
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]


#lib.dn_test()
