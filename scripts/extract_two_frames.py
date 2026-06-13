import cv2
import os

def main():
    video_path = "eval_run.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return
        
    output_dir = "extracted_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Extract 1s and 15s frames
    for sec in [1, 15]:
        frame_idx = int(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"frame_{sec}s.jpg")
            cv2.imwrite(out_path, frame)
            print(f"Extracted {out_path} successfully.")
        else:
            print(f"Error: Could not extract frame at {sec}s.")
            
    cap.release()

if __name__ == "__main__":
    main()
