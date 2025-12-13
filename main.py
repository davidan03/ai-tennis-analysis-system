from utils import read_video, save_video

def main():
    input_video_path = "inputs/input_video.mp4"
    video_frames = read_video(input_video_path)

    output_video_path = "outputs/output_video.avi"
    save_video(video_frames, output_video_path)

if __name__ == "__main__":
    main()