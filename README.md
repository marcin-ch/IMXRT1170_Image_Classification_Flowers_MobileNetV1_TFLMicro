# Overview
This is an example of image classification on i.MXRT1170 evaluation kit ([MIMXRT1170-EVK](https://www.nxp.com/design/development-boards/i-mx-evaluation-and-development-boards/i-mx-rt1170-evaluation-kit:MIMXRT1170-EVK)).

It is heavily based on NXP tutorial [Getting Started with TensorFlow Lite for Microcontrollers on i.MX RT](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/Getting-Started-with-TensorFlow-Lite-for-Microcontrollers-on-i/ta-p/1124103). Following the description of the tutorial:
> it covers how to take an existing TensorFlow image classification model named Mobilenet, and re-train it to categorize images of flowers. This is known as transfer learning. This updated model will then be saved as a TensorFlow Lite file. By using that file with the TensorFlow Lite for Microcontrollers inference engine that is part of NXPs [eIQ](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ) package, the model can be ran on an i.MXRT embedded device. A camera attached to the board can then be used to look at photos of flowers and the model will determine what type of flowers the camera is looking at. These same steps could then be used for classifying other types of images too.
>
> This lab can also be used without a camera+LCD, but in that scenario the flowers images will need to be converted to a C array and loaded at compile time.  

**Main document** is [eIQ TensorFlow Lite for Microcontrollers Lab for RT1170 - With Camera.pdf](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/Getting-Started-with-TensorFlow-Lite-for-Microcontrollers-on-i/ta-p/1124103?attachment-id=118735), which is also available in */doc* folder.

# Hardware
* [MIMXRT1170-EVK](https://www.nxp.com/design/development-boards/i-mx-evaluation-and-development-boards/i-mx-rt1170-evaluation-kit:MIMXRT1170-EVK) with attached:
  * [RK055HDMIPI4M](https://www.nxp.com/design/development-boards/i-mx-evaluation-and-development-boards/i-mx-rt1170-evaluation-kit:MIMXRT1170-EVK#buy) 720x1280 LCD display
  * camera module (based on paper list attached to kit, it is 2592x1944 camera module image sensor with Huatian technology)

# Software
* MCUXpresso IDE v11.4.0 [Build 6224] [2021-07-15]
* SDK_2.x_EVK-MIMXRT1170 version 2.10.1

# Workflow
## Retrain exisiting model
NXP suggests installing Python, TensorFlow, TensorFlow Lite Model Maker, Vim and all required tools on your local disc drive, however seems it is better to use **Google Colab** which allows you to write and execute Python in your browser.

All the steps required to retrain existing model are covered in this notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcin-ch/IMXRT1170_Image_Classification_Flowers_MobileNetV1_TFLMicro/blob/master/doc/IMXRT1170_Image_Classification_Flowers_MobileNetV1_TFLiteMicro.ipynb)

Hit the button and then `Ctrl`+`F9` (**Run All**).

This notebook is also available in */doc* folder.

Once notebook is executed, you get 3 files:
* retrained model *flower_model.tflite*
* converted model *flower_model.h*
* labels *flowers_labels.txt*

Save them to your favourite directory at your local disc drive. 

## Update an example from SDK to recognize flowers
1. Import SDK to your MCUXpresso
2. From **QuickStart Panel** import SDK example *tensorflow_lite_micro_label_image*
    * in **Project options** change **SDK Debug Console** to **UART** to use external debug console via UART (should be checked by default)
3. In **Project Explorer** change project's name to, for example, *evkmimxrt1170_SDK2101_Image_Class_Flowers_TFLM*
4. Move following files to */doc* folder
    * *flower_model.tflite*
    * *flowers_labels.txt*
5. Move *flower_model.h* to */source/model* folder
6. In MCUXpresso, open */source/model/flower_model.h* and change it from:
    ```c
    unsigned char flower_model_tflite[] = {
    ```
    to:
    ```c
    #include <cmsis_compiler.h>

    #define MODEL_NAME "mobilenet_v1_0.25_128_flower"
    #define MODEL_INPUT_MEAN 127.5f
    #define MODEL_INPUT_STD 127.5f

    const char flower_model_tflite[] __ALIGNED(16) = {
    ```
7. In **Project Explorer** copy and paste */source/labels.h* and rename it to *flowers_labels.h*. Update then array with labels according to */doc/flower_labels.txt* to get as follows: 
    ```c
    static const char* labels[] = {
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
    };
    ```
8. In */source/model/model.cpp* make following changes:
    * add ops resolver that supports all the operands used by retrained model. Following [Get started with microcontrollers (section 1. Include the library headers)](https://www.tensorflow.org/lite/microcontrollers/get_started_low_level#1_include_the_library_headers) it provides the operations used by interpreter to run the model
    ```cpp
    #include "tensorflow/lite/micro/all_ops_resolver.h"
    ```
    * comment out original model from SDK example and add retained model:
    ```cpp
    #include "flower_model.h"
    // #include "model_data.h"
    ```
    * adjust an area of memory to use for input, output, and intermediate arrays, retrained model is larger than original model from SDK example
    ```cpp
    constexpr int kTensorArenaSize = 800000;
    // constexpr int kTensorArenaSize = 512 * 1024;
    ```
    * change the model name to the array name from */source/model/flower_model.h*
    ```cpp
    s_model = tflite::GetModel(flower_model_tflite);
    // s_model = tflite::GetModel(model_data);
    ```
    * change operations resolver, following [Get started with microcontrollers (section 6. Instantiate operations resolver)](https://www.tensorflow.org/lite/microcontrollers/get_started_low_level#6_instantiate_operations_resolver) this will be used by the interpreter to access the operations that are used by the model. The `AllOpsResolver` loads all of the operations available in TensorFlow Lite for Microcontrollers which uses a lot of memory. Since a given model will only use a subset of these operations it is recommended that real world applications load only the operations that are needed. This is done using different class `MicroMutableOpResolver`.
    ```c
    tflite::AllOpResolver micro_op_resolver;
    // tflite::MicroOpResolver &micro_op_resolver = MODEL_GetOpsResolver(s_errorReporter);
    ```
9. In */source/model/output_postproc.cpp* make following changes:
    * comment out labels from original SDK example and add labels for retrained model:
    ```cpp
    #include "flowers_labels.h"
    // #include "labels.h"
    ```
10. In **QuickStart Panel** hit **Debug** and, after flashing the board, application should recognize 5 types of flowers.
11. By default, **Build Configurations** is set to **Debug** and it leads to inference time approximately 3,9 seconds which is quite high. To enable high compile optimizations and in result significantly decrease inference time of TensorFlow Lite for Microcontrollers project, change **Build Configurations** to **Release** (**Project** -> **Build Configurations** -> **Set Active** -> **Release**) and hit **Debug** again. Finally, inference time is approximately 550 ms.
12. Use terminal such as Tera Term to get bit more detailed output from an application, especially inference time.

# Summary
Inference time is approximately 3,9 seconds, when **Build Configurations** is set to **Debug**. When it is changed to **Release** inference time dramatically drops to approximately 550 ms.

To test the application, you can use prepared by NXP [Flowers.pdf](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/Getting-Started-with-TensorFlow-Lite-for-Microcontrollers-on-i/ta-p/1124103?attachment-id=14022) document with few flowers. You can find it also in */doc* folder.

Model performance is bit poor, [eIQ TensorFlow Lite for Microcontrollers Lab for RT1170 - With Camera.pdf](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/Getting-Started-with-TensorFlow-Lite-for-Microcontrollers-on-i/ta-p/1124103?attachment-id=118735) says that:
> this is because the model was trained on the specific images, and the camera output won’t quite match those images, thus leading to decreased accuracy. This is why in a production system, it is best to train on the actual data generated by the camera.

What is also important, following [eIQ TensorFlow Lite for Microcontrollers Lab for RT1170 - With Camera.pdf](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/Getting-Started-with-TensorFlow-Lite-for-Microcontrollers-on-i/ta-p/1124103?attachment-id=118735):
> You also may notice that even when the camera is pointed at random objects, it still attempt to categorize them as a flower type. This is because when the model was retrained, it was only retrained on flower images. The concept of any other type of object is unknown to the model, so it attempts to classify everything as one of the 5 types that it does know.

## How to use this repo (source code)
1. Clone the repo or download as `*.zip` to your local disc drive
    * when clonning please use below command:
    ```console
    git clone https://github.com/marcin-ch/IMXRT1170_Image_Classification_Flowers_MobileNetV1_TFLMicro.git
    ```
2. Open MCUXpresso, you will be asked for choosing existing or creating new workspace, I recommend creating new workspace for testing purposes
3. From `QuickStart Panel` choose `Import project(s) from file system` and then select either unpacked repo (in case you clonned the repo) or zipped repo (in case you downloaded the archive)
4. Make sure `Copy projects into workspace` in `Options` is checked
5. Hit `Finish`
6. Select imported project in `Project Explorer` and hit `Debug` in `QuickStart Panel`, the application should be up and running
7. You can now remove clonned or downloaded repo, as it now exists in your workspace

## How to use this repo (binary file)
If you just want to check how the project looks like running on the board, you can flash binary files available in */doc* folder:
* *evkmimxrt1170_SDK2101_Image_Class_Flowers_TFLM_Debug.bin*
* *evkmimxrt1170_SDK2101_Image_Class_Flowers_TFLM_Release.bin*
 
 As i.MXRT1170 evaluation kit enumerates as MSD (Mass Storage Device) when connected to PC through USB cable, you can simply drag-n-drop binary file to your board. Wait few moments when flashing is in progress, reset the board and you should see application working.