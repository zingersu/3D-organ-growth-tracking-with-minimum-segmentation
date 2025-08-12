3D organ growth tracking with minimum segmentation
=====
Prerequisites
------
In the standard mode, we use 3D-NOD framework as the pre-trained network and apply it as a new organ detection module. For details on its specific implementation, please refer to https://github.com/zingersu/3D-New-Organ-Detection-in-Plant-Growth-from-Spatiotemporal-Point-Clouds
After identifying new and old organs, all codes run under the pytorch version, and its corresponding configurations are as follows:<br>
* The 3D-OGT framework runs under Windows 11<br>
* Current code execution environment:<br>
    * Python == 3.8.20<br>
    * Pytorch == 2.1.2<br>
    * CUDA == 12.1<br>
    * Pandas == 2.0.3<br>
    * Scikit-learn == 1.3.2<br>

Introduction
------
To monitor the growth and structural changes of crop organs, dynamic plant phenotyping based on time-series point clouds has become a cutting-edge research topic. However, existing organ tracking methods based on crop time-series point clouds either rely on complete organ instance segmentation results or lack real-time performance in capturing spatiotemporal correlations among organs. These limitations significantly hinder the development of dynamic crop phenotyping.<br>
<br>
3D-OGT is a framework capable of performing continuous organ tracking throughout the entire growth sequence with only the minimal segmentation information. Our framework can automatically propagate organ labels from the previous moment's crop point cloud to the subsequent point cloud, while completing organ segmentation and tracking on multiple crop growth sequences. The framework can recognize and track new organs, mature organs, and even suddenly disappeared organs. Experimental results on a spatiotemporal point cloud dataset demonstrate that the 3D-OGT framework achieves satisfactory organ tracking performance.<br>
<p align="center">
  <strong><em>The overall framework of 3D-OGT. (a) is the step for new organ detection of point cloud based on the pre-trained 3D-NOD network; (b) implements inhomogeneous down-sampling; (c) employs semi-supervised learning for organ label broadcasting; (d) performs the label propagation and refinement. </em></strong>
</p>

Quick Start
------
This project contains six folders.<br>
folder <strong>[code]</strong> contains the complete implementation of the 3D-OGT framework<br>
folder <strong>[norm_GT_fps]</strong> contains all point cloud files with complete and precise instance labels, serving as the foundation for calculating quantitative tracking metrics<br>
folder <strong>[instance_new_organ(3D-NOD)]</strong> and folder <strong>[instance_old_organ(3D-NOD)]</strong> comprise the outputs of the 3D-NOD framework, which is uesd as a pre-trained network for detecting new organs. Then the complete plant point cloud is divided into two categories: new organs and old organs.<br>
folder <strong>[instance_new_organ(by human)]</strong> and folder <strong>[instance_old_organ(by human)]</strong> contain their manually curated counterparts, where human annotation replaces the pre-trained network in detecting new organs before dividing them into the same two categories.<br>
<br>
<strong>Note:</strong> When running the code, if standard_mode is True, it refers to the standard 3D-OGT using the pre-trained 3D-NOD network; conversely, if standard_mode is False, it represents the control group employing the fully manual new organ detection module.<br>
<br>
