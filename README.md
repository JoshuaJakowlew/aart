# Aart
Love ascii-art? Me too! This utility converts images and video to beautiful and colorful ascii art.
No explanations, just look at the example. I used default 16-color palette from cmd.exe

![Image](aart/test.jpg) 
![Image](aart/new-ascii-art.png) 

It's simple but beautiful. You can convert even videofiles.
[This is](https://youtu.be/HAmjZi_CUzo) sample video converted with custom palette (not the best one).

Here is command-line syntax
```
Usage: aart.exe [params]

        -?, -h, --help, --usage (value:true)
                print this message
        --charmap, --chr (value:charmap.png)
                charmap
        --cie94 (value:true)
                use more precise but more expensive algorithm
        --clr, --colormap (value:colormap.png)
                colormap
        --colors (value:16)
                number of colors in palette mode
        -i
                input file
        --mode (value:image)
                render mode [image,video,ansi,palette]
        -o
                output file
        --quantization (value:dominant)
                color quantization algorithm [kmean,dominant]
        --use_cuda (value:false)
                use cuda backend if possible
```
Aart includes sample palette and mediafiles that were used for testing.

# Recommendations and known problems
When working with videofiles, there can be problems with video encoders/decoders. Aart uses OpenCV as a backend.
In my machine OpenCV writes some random codec-related errors. Nevertheless, conversion is successful.

# Build
Aart depends on OpenCV. You can provide headers and `.lib`s by yourself or use vcpkg.
If using vcpkg, then type the following commad:
```vcpkg install opencv4[contrib,cuda,dnn,ffmpeg,jpeg,opengl,png,tiff,webp]:x64-windows```

Aart works currently on Windows only. But you can easily port it to any platform that supports OpenCV - just create CMake project
or build VS solution with linux as target OS.
