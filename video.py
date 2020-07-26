import cv2
import numpy as np
import pickle
import os

# MPEG-1 users different quantization matrix for Luminance and
# Chrominance but we are using just the Luminance Matrix
# here for ease
QUANTIZATION_MATRIX = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])


# This class reads the raw video and converts it into the RBG format
# which we can work with, reading raw video isn't straight forward because
# it doesn't have any headers to describe the video like in most compressed
# formats
class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 3 // 2)
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), int(self.width))

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
            # print(np.shape(yuv))
        except Exception as e:
            print(str(e), '#')
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        # print(np.shape(bgr))
        return ret, bgr


# This function straight away writes the video as MPEG-4 to show what
# compression we are getting over raw video
def write_mpeg4(frames, size, name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(name + '.mp4', fourcc, 30, (size[1], size[0]))
    for frame in frames:
        video_out.write(frame)


# Zig Zag Run Length Coding of a 8x8 block
def zig_zig_rlc(block):
    coding = []
    # block is assumed to be 8x8
    (j, i) = (0, 0)
    direction = 'r'  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
    code = 0
    length = 0
    for _ in range(64):
        if code == block[j][i]:
            length += 1
        else:
            if length > 0:
                coding.append(code)
                coding.append(length)
            code = block[j][i]
            length = 1

        if direction == 'r':
            i += 1
            if j == 7:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'dl':
            i -= 1
            j += 1
            if j == 7:
                direction = 'r'
            elif i == 0:
                direction = 'd'
        elif direction == 'd':
            j += 1
            if i == 0:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'ur':
            i += 1
            j -= 1
            if i == 7:
                direction = 'd'
            elif j == 0:
                direction = 'r'

    if length > 0:
        coding.append(code)
        coding.append(length)

    return coding


# Convert the Zig Zag Code to 8x8 Blocks
def zigzag2blocks(zigzag_pattern):
    blocks = []
    (j, i) = (0, 0)
    direction = 'r'  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
    length = np.shape(zigzag_pattern)[0]
    # print(length)
    block = np.zeros((8, 8))
    for c in range(int(length/2)):
        code = zigzag_pattern[2*c]
        leng = zigzag_pattern[2*c + 1]

        for _ in range(0, leng):

            if j > 7 or i > 7:
                (j, i) = (0, 0)
                direction = 'r'  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
                blocks.append(block)
                block = np.zeros((8, 8))

            block[j][i] = code
            if direction == 'r':
                i += 1
                if j == 7:
                    direction = 'ur'
                else:
                    direction = 'dl'
            elif direction == 'dl':
                i -= 1
                j += 1
                if j == 7:
                    direction = 'r'
                elif i == 0:
                    direction = 'd'
            elif direction == 'd':
                j += 1
                if i == 0:
                    direction = 'ur'
                else:
                    direction = 'dl'
            elif direction == 'ur':
                i += 1
                j -= 1
                if i == 7:
                    direction = 'd'
                elif j == 0:
                    direction = 'r'
    if j > 7 or i > 7:
        blocks.append(block)
    return blocks


# Function to compress a single channel
def single_channel_compression(img, scale_factor=1):
    height = img.shape[0]
    width = img.shape[1]

    image_blocks = []
    # split the image into 8x8 blocks
    for j in range(0, height, 8):
        for i in range(0, width, 8):
            image_blocks.append(img[j:j+8, i:i+8])


    # dct for each block
    image_blocks_dct = [cv2.dct(image_block) for image_block in image_blocks]

    # quantization of each block
    reduced_blocks = [np.int8(np.round(dct_block / (QUANTIZATION_MATRIX * scale_factor)))
                      for dct_block in image_blocks_dct]

    return reduced_blocks


# Function decompress a single channel
def single_channel_decompression(blocks, size, scale_factor=1):
    img = np.zeros(size)

    # reverse quantization
    unreduced_blocks = [block * (QUANTIZATION_MATRIX * scale_factor)
                        for block in blocks]

    # IDCT of each block
    image_blocks = [cv2.idct(np.float32(block)) for block in unreduced_blocks]

    height = size[0]
    width = size[1]

    m = 0
    for j in range(0, height, 8):
        for i in range(0, width, 8):
            img[j:j+8, i:i+8] = image_blocks[m]
            m += 1

    return img


# Function to compress one frame
def compress_image(image, size, prev_image=None, scale_factor=1):

    # print(image[0:4, 0:4])
    img = np.float32(image)
    # Chance to YCC color space
    img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # User previous frame to reduce information
    if prev_image is not None:
        prev_image = np.float32(prev_image)
        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2YCrCb)
        img_ycc = img_ycc - prev_image
    compressed_blocks = []

    for i in range(3):
        compressed = single_channel_compression(img_ycc[:, :, i],
                                                scale_factor)
        compressed_blocks = compressed_blocks + compressed

    # Convert 8x8 blocks into zigzag codes
    zigzag_pattern = []
    for block in compressed_blocks:
        zigzag_block = zig_zig_rlc(block)
        zigzag_pattern += zigzag_block

    zigzag_pattern = np.int8(zigzag_pattern)
    return zigzag_pattern


# Function to decompress one frame
def decompress_image(compressed_blocks, size, prev_frame=None, scale_factor=1):
    channel_blocks = int(size[0] * size[1] / 64)
    recreate_image = np.zeros((size[0], size[1], 3))

    for i in range(3):
        uncompressed = single_channel_decompression(
            compressed_blocks[channel_blocks*i: channel_blocks*(i+1)],
            size, scale_factor)
        recreate_image[:, :, i] = uncompressed
    recreate_image = np.float32(recreate_image)

    if prev_frame is not None:
        prev_frame = np.float32(prev_frame)
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YCrCb)
        recreate_image = recreate_image + prev_frame

    image = cv2.cvtColor(recreate_image, cv2.COLOR_YCrCb2BGR)
    image[image < 0] = 0
    image[image > 255] = 255
    image = np.uint8(image)
    return image


# Compress a Video
def compress_video(frames, size, file_name, scale_factor=1):

    complete_pattern = []
    frame_number = np.shape(frames)[0]
    for i in range(frame_number):
        print('Compression Frame', i)
        if i > 0:
            zigzag_pattern = compress_image(frames[i], size,
                                            None, scale_factor)
        else:
            zigzag_pattern = compress_image(frames[i], size,
                                            None, scale_factor)

        complete_pattern = np.concatenate(
            (complete_pattern, zigzag_pattern))
        print(np.shape(complete_pattern))
        if i == 60:
            break

    complete_pattern = np.int8(complete_pattern)

    with open(file_name, 'wb') as f:
        pickle.dump(complete_pattern, f)


# Decompress the Video
def decompress_video(file_name, size, scale_factor=1):
    channel_blocks = int(size[0] * size[1] / 64)
    image_blocks = channel_blocks * 3

    with open(file_name, 'rb') as f:
        complete_pattern = pickle.load(f)

    video_blocks = zigzag2blocks(complete_pattern)

    pattern_length = np.shape(video_blocks)[0]
    total_frames = int(pattern_length / image_blocks)
    frames = []

    for i in range(total_frames):
        if i > 0:
            frame = decompress_image(
                    video_blocks[image_blocks*i: image_blocks*(i+1)],
                    size, None, scale_factor)
        else:
            frame = decompress_image(
                video_blocks[image_blocks * i: image_blocks * (i + 1)],
                size, None, scale_factor)

        frames.append(frame)

    for frame in frames:
        cv2.imshow('decompressed', frame)
        cv2.waitKey(30)

    return frames


if __name__ == "__main__":
    filename = "data/akiyo_cif.yuv"
    # filename = "data/stefan_cif.yuv"
    size = (288, 352)

    cap = VideoCaptureYUV(filename, size)
    frames = []
    while 1:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    for frame in frames:
        cv2.imshow('raw', frame)
        cv2.waitKey(30)

    video_name = 'compressed_video_file'
    scale_factor = 1

    print("***Starting Compression****")
    compress_video(frames, size, video_name, scale_factor)

    print("***Starting Decompression****")
    video = decompress_video(video_name, size, scale_factor)

    print("***Starting Writing to MPEG4****")
    write_mpeg4(frames, size, 'akioyo')

    directory = 'test/'
    try:
        os.mkdir(directory)
    except:
        pass

    m = 1
    for frame in frames:
        cv2.imwrite(directory + 'frame_raw_' + str(m) + '.png', frame)
        m += 1

    m = 1
    for frame in video:
        cv2.imwrite(directory + 'frame_compressed_' + str(m) + '.png', frame)
        m += 1

