%%cl_kernel

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void box_blur(read_only image2d_t image, write_only image2d_t output)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    int width = get_image_width(image);
    int height = get_image_height(image);
    
    uint4 sum = (uint4)(0, 0, 0, 0);
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = clamp(coord.x + dx, 0, width - 1);
            int ny = clamp(coord.y + dy, 0, height - 1);
            int2 ncoord = (int2)(nx, ny);
            sum += read_imageui(image, sampler, ncoord);
        }
    }
    
    uint4 avg = sum / 9;
    write_imageui(output, coord, avg);
}