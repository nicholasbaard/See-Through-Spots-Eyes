import cv2

# Initialize video capture
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.80.3:21554/media/video1')  # Use 0 for default camera, or provide a video file path

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('output/video/output.mp4', fourcc, 20.0, (640, 480))

frame_number = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    
    if ret:
        # Write the frame
        out.write(frame)
    
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        
        
        # Save frame with a frame number
        # frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.imwrite(f'output/frames/frame_{frame_number}.jpg', frame)
        frame_number += 1
    
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
