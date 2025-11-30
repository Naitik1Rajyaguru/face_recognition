# Setup & Usage Instructions

1. **Install Dependencies**
   Install the required Python packages:

   ```bash
   pip install -r req.txt
   ```

2. **Add Images for Tracking**
   Place the images you want to track inside the `Images` folder.

3. **Configure Tracking in `main.py`**

   - Update the file paths and names of the images in `main.py` to enable tracking.
   - Update the camera settings according to your setup:

     - `0` for the system default camera
     - Or provide the URL of an IP camera (e.g., an Android webcam app).

4. **Run the Application**

   ```bash
   python main.py
   ```
