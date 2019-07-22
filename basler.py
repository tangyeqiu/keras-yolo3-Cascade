import os
from imageio import get_writer
import cv2
from pypylon import pylon

###############################################################################
# connecting to the first available camera

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabbing Continuously (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to OpenCV bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
###############################################################################

out = get_writer(
    'C:/Users/tag2sgh/Documents/GitHub/keras-yolo3/test_data/output-filename.mp4',
    fps=30,  # FPS is in units Hz; should be real-time.
    codec='libx264rgb',  # When used properly, this is basically
    # "PNG for video" (i.e. lossless)
    quality=None,  # variable compression
    pixelformat='rgb24',  # keep it as RGB colours
    ffmpeg_params=[  # compatibility with older library versions
        '-preset',  # set to faster, veryfast, superfast, ultrafast
        'fast',  # for higher speed but worse compression
        '-crf',  # quality; set to 0 for lossless, but keep in mind
        '11'  # that the camera probably adds static anyway
    ]
)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        frame = image.GetArray()

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", frame)
    # if isOutput:
    #     out.write(result)
    out.append_data(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
