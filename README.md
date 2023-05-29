# Defects Detection in SEM Images
Given two SEM (Scanning Electron Microscope) images, a query (“inspected”) image and a train (“reference”) image, the goal is to detect defect patterns such as bridge, break and line collapses as well as blotches. Note that only classical computer vision techniques are used here.

## Assumptions
### Query and Train images are not aligned
In other words, there are areas of the query image that are not represented in the train image. Therefore, any defect that may be present in these non-overlapping regions will not be detected using the information provided by the train image. 

Example:

<div align=center>
  <img src="input/case3_inspected_image.tif" width="480" style="margin:20">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="input/case3_reference_image.tif" width="480" style="margin:20"> 
</div>
<div align=center>
 an example of the query image 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 the corresponding train image
</div>

### Statistical modeling, aka ML, is out of scope
Instead, I solely employ classical computer vision techniques to address the problem at hand. Consequently, this solution lacks generalizability and should only be considered as a foundational framework.
It is worth mentioning that contemporary methods can tackle this type of problem without the need for training images. ML algorithms, based on statistical approaches, offer significantly enhanced robustness and can achieve near-perfect accuracy down to the pixel level.

 
### Defect types considered

1. Radial blobs/blotches
2. Elongated cracks/pits
3. Hair-like: thin (often 1-pixel wide), elongated features that resemble strands of hair.
<br>
Here are some examples:
<br>
<div align=center><img src="images/defect_types.tif" width="480" style="margin:20">
</div>
<div align=center>
an example of defect types
</div>

## Approach
1.	Apply image registration by template matching to align the two images.
2.	Calculate the difference between the grayscale versions of the two aligned images.
3.	Detect defects by type:
    a.	Look for radial blobs
        i.	Sharpen the difference image
        ii.	Threshold it
        iii.	Open it (i.e. erode and dilate)
        iv.	Return the resulting mask
    b.	Look for elongated cracks/pits
        i.	Threshold the difference image
        ii.	Open it (i.e. erode and dilate)
        iii.	Return the resulting mask
    c.	Look for hair-like defects
        i.	Threshold the difference image
        ii.	Find contours in it
        iii.	On a clean mask, draw only the “hair” shaped contours
        iv.	Return a mask with these contours
4.	Aggregate the results into a single binary mask
5.	Display/save it


## Results
Here are a few outcomes generated by my code, utilizing the OpenCV library.

Result pair 1:
<div align=center>
  <img src="output/case1_inspected_image.tif" width="340" style="margin:20">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="output/case1_inspected_image_defects_mask.tif" width="340" style="margin:20"> 
</div>

Result pair 2:
<div align=center>
  <img src="output/case2_inspected_image.tif" width="480" style="margin:20">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="output/case2_inspected_image_defects_mask.tif" width="480" style="margin:20"> 
</div>

Result pair 3 (no defects):
<div align=center>
  <img src="output/case3_inspected_image.tif" width="480" style="margin:20">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="output/case3_inspected_image_defects_mask.tif" width="480" style="margin:20">
</div>

## Usage
```
git clone https://github.com/giyorahy/defect-detection-in-sem-images.git
cd defect-detection-in-sem-images/src
pip install -r requirements.txt 
python main.py
```
