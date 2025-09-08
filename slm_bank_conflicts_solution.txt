typedef struct
{
    int a;
    int buffer_data[31];
    int padding; // this pads the structure to uneven number
} SomeInfo;