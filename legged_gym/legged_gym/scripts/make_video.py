import cv2
import os
from tqdm import tqdm
import functools
image_folder = '../../logs/push_door_cyber/exported/frames/20000'  # directory to the pictures
video_name = '../../videos/output_pushdoor_mlp_right_80_20000.mp4'  # name for the video

# image_folder = './logs/fixstand_cyberdog2/exported/frames'  # directory to the pictures
# video_name = 'output_fixdance.mp4'  # name for the video

#ffmpeg -r 50 -i ./logs/dance_cyber/exported/frames/%d.png -pix_fmt yuv420p output_dancing0.mp4

def comp(A,B):
    a=eval(A[:-4])
    b=eval(B[:-4])
    print("cmp a,b",a,b,"a<b?",a<b)
    if a<b:
        return 1
    return 0

def f(A):
    return int(A[:-4])

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#images.sort(key=functools.cmp_to_key(compare))
images=sorted(images,key=f)

# images=['345.png','5348527.png','1.png']
# b=sorted(images,key=functools.cmp_to_key(comp))
# print("images=",b)

# c=sorted(images,key=functools.cmp_to_key(comp2))
# print("images=",c)

# d=sorted(images,key=f)
# print("d=",d)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0x7634706d, 50, (width,height))

for image in tqdm(images):
    #print(image)
    frame = cv2.imread(os.path.join(image_folder, image))
    video.write(frame)

cv2.destroyAllWindows()
video.release()
