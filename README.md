# AR-Tag-Detection-and-Tracking
### Pipeline for tag detection and decoding
1. Fast fourier transform of the thresholded image
<p float="left">
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/thresh.png" width="350" />
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/fft_back.png" width="350" />
</p>

2. Masking with a dynamic circular mask centered at the mean of white pixels, eliminating pixels noise.
<p float="left">
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/mask_viz.png" width="350" align="centered"/>
</p>
3. Corner detection and obtaining extreme corners in x and y directions.
<p float="left">
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/all_corners.png" width="350" />
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/outer_extreme_corners.png" width="350" />
</p>
4. Cropping image and finding inner extreme corners in x and y directions.
<p float="left">
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/inner_corners.png" width="350" />
</p>
5. Warping the image to decode tag.
<p float="left">
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/warped.png" width="350" align="centered" />
</p>
6. Result
<p float="left">
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/tag_id_on_frame.jpg" width="400" align="centered" />
</p>

### Superposing an image onto the tag
<p float="left">
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/testudo.gif" width="500" align="centered" />
</p>

### Projecting an AR cube onto the tag

<p float="left">
	<img src="https://github.com/niteshjha08/AR-Tag-Detection-and-Tracking/blob/main/media/img/cube.gif" width="500" align="centered" />
</p>