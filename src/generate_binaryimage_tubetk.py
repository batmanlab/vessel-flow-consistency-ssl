'''
Given the .tre files and MRA image, generate the corresponding binary image
'''
from data_loader.tubetkdataset import TubeTKDataset
import os, sys
from os import path as osp
import numpy as np
import itk
from itk import TubeTK as ttk
from itkwidgets import view

PixelType = itk.F
Dimension = 3
ImageType = itk.Image[PixelType, Dimension]

# Read tre file
TubeFileReaderType = itk.SpatialObjectReader[Dimension]

def main():
    ds = TubeTKDataset('/ocean/projects/asc170022p/rohit33/TubeTK',)
    outputdir = '/ocean/projects/asc170022p/rohit33/TubeTKoutput/'

    # For each file, generate binary label and save it
    for i, (mra, tre) in enumerate(zip(ds.allmra, ds.alltre)):

        tubeFileReader = TubeFileReaderType.New()
        tubeFileReader.SetFileName(tre)
        tubeFileReader.Update()
        tubes = tubeFileReader.GetGroup()

        # Get image too
        tmpImageType = itk.Image[PixelType, Dimension]
        tmpImageReaderType = itk.ImageFileReader[tmpImageType]
        tmpImageReader = tmpImageReaderType.New()
        tmpImageReader.SetFileName(mra)
        tmpImageReader.Update()
        img = tmpImageReader.GetOutput()

        # Get binary
        TubesToImageFilterType = ttk.ConvertTubesToImage[tmpImageType]
        tubesToImageFilter = TubesToImageFilterType.New()
        tubesToImageFilter.SetUseRadius(True)
        tubesToImageFilter.SetTemplateImage(tmpImageReader.GetOutput())
        tubesToImageFilter.SetInput(tubes)
        tubesToImageFilter.Update()

        outputImage = tubesToImageFilter.GetOutput()
        outputPath = osp.join(outputdir, 'gt_{}.npy'.format(i))
        with open(outputPath, 'wb') as fi:
            np.save(fi, outputImage)
        print("Saved to {}".format(outputPath))


if __name__ == "__main__":
    main()
