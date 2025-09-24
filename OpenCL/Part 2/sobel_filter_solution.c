%%cl_kernel

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void sobel_filter(read_only image2d_t image, 
                           const __global float* my_filter, 
                           write_only image2d_t output)
{    
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    int2 image_size = (int2)(get_image_width(image), get_image_height(image));
    
    float acc = 0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            int sx = clamp(coord.x + x, 0, image_size.x - 1);
            int sy = clamp(coord.y + y, 0, image_size.y - 1);
            uint4 p = read_imageui(image, sampler, (int2)(sx, sy));
            
            float luma = (p.x + 2.0 * p.y + p.z) * 0.25;
            float weight = my_filter[3 * (y + 1) + (x + 1)];
            acc += weight * luma;
        }
    }    
    
    acc = fabs(acc);
    acc = fmin(acc, 255.0f);
    
    write_imageui(output, coord, (uint4)(acc, 0, 0, 0));
}