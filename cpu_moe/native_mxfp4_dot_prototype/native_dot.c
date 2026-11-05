/* Independent prototype: native consecutive FP4 bytes + separate E8M0 scales.
 * No persistent repack, no allocation, no changes to installed kt-kernel. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static const int8_t lut[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static inline float half_scale(uint8_t e) {
    uint32_t bits=e<2 ? (uint32_t)0x00200000<<e : (uint32_t)(e-1)<<23;
    float f; memcpy(&f,&bits,4); return f;
}
static inline float activation_scale(const uint8_t *q) {
    __fp16 h; memcpy(&h,q,2); return (float)h;
}
static inline int32x4_t block_dot(const uint8_t *w,const uint8_t *q) {
    uint8x16_t b=vld1q_u8(w);
    int8x16_t table=vld1q_s8(lut);
    int8x16_t lo=vqtbl1q_s8(table,vandq_u8(b,vdupq_n_u8(15)));
    int8x16_t hi=vqtbl1q_s8(table,vshrq_n_u8(b,4));
    /* Original K order. Matches the reference's two half-block dot groups. */
    int8x16_t a=vzip1q_s8(lo,hi),c=vzip2q_s8(lo,hi);
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0),a,vld1q_s8((const int8_t*)q+2)),
                     c,vld1q_s8((const int8_t*)q+18));
}
int native_dot(int k,const uint8_t *w,const uint8_t *scales,const uint8_t *q,float *out) {
    if(k<=0 || k%32 || !w || !scales || !q || !out) return -1;
    int nb=k/32;
    /* 255 is reserved for NaN in E8M0. Reject explicitly. */
    for(int b=0;b<nb;b++) if(scales[b]==255) return -2;
    float32x4_t s0=vdupq_n_f32(0),s1=vdupq_n_f32(0);
    int b=0;
    for(;b+1<nb;b+=2) {
        s0=vfmaq_n_f32(s0,vcvtq_f32_s32(block_dot(w+16*b,q+34*b)),half_scale(scales[b])*activation_scale(q+34*b));
        s1=vfmaq_n_f32(s1,vcvtq_f32_s32(block_dot(w+16*(b+1),q+34*(b+1))),half_scale(scales[b+1])*activation_scale(q+34*(b+1)));
    }
    float result=vaddvq_f32(vaddq_f32(s0,s1));
    if(b<nb) {
        int sum=0;
        const int8_t *a=(const int8_t*)(q+34*b+2);
        for(int j=0;j<16;j++) sum+=lut[w[16*b+j]&15]*a[2*j]+lut[w[16*b+j]>>4]*a[2*j+1];
        result += (half_scale(scales[b])*activation_scale(q+34*b))*sum;
    }
    *out=result; return 0;
}
