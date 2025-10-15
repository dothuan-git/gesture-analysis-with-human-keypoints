import cv2
import pandas as pd
import utils as utils

# ——— CONFIG ———
VIDEO_PATH = 'assets/GX010016_1080_120fps.MP4'
CSV_PATH   = 'data/merged_hands.csv'
FLIP_FRAME = True  # set False if you logged without flipping
SAVE_FRAMES = True  # save annotated frames to disk

# optional: change radius/thickness
POINT_RADIUS    = 3
LEFT_COLOR_BGR  = (0, 255, 0)   # green
RIGHT_COLOR_BGR = (0, 0, 255)   # red

def main():
    output_dir = f'data/{VIDEO_PATH.split("/")[-1].split(".")[0]}'

    # 1) load CSV and index by frame
    df = pd.read_csv(CSV_PATH)
    df.set_index('frame', inplace=True)

    # 2) open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video `{VIDEO_PATH}`")
        return

    # 3) grab one frame to get size
    ret, sample = cap.read()
    if not ret:
        print("Error: could not read first frame")
        return
    if FLIP_FRAME:
        sample = cv2.flip(sample, 1)
    h, w = sample.shape[:2]

    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    # 4) process loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        # draw if we have any data for this frame
        if frame_idx in df.index:
            rows = df.loc[[frame_idx]] if isinstance(df.loc[frame_idx], pd.Series) else df.loc[frame_idx]
            if isinstance(rows, pd.Series):
                rows = [rows]
            # Fix: use .iterrows() if rows is a DataFrame
            if isinstance(rows, pd.DataFrame):
                rows = [row for _, row in rows.iterrows()]
            for row in rows:
                # for each hand
                for hand_name, color in [('left', LEFT_COLOR_BGR),
                                         ('right', RIGHT_COLOR_BGR)]:
                    # 21 landmarks each
                    for i in range(21):
                        x_norm = row.get(f'{hand_name}_{i}_x')
                        y_norm = row.get(f'{hand_name}_{i}_y')
                        # skip missing points
                        if pd.notnull(x_norm) and pd.notnull(y_norm):
                            x_px = int(x_norm * w)
                            y_px = int(y_norm * h)
                            cv2.circle(frame,
                                       (x_px, y_px),
                                       POINT_RADIUS,
                                       color,
                                       thickness=-1)

        # Draw frame index
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Save annotated frame
        if SAVE_FRAMES:
            frames_dir = f'{output_dir}/frames_check'
            utils.create_folder_if_not_exist(frames_dir)
            cv2.imwrite(f'{frames_dir}/{frame_idx}.jpg', frame)

        cv2.imshow('Hand Landmarks Overlay', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
