/* Isolated native MXFP4 full MoE prototype. Borrows immutable weight pointers.
 * Scratch scales with H/I only; no expert weight repack or persistent copy. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int native_dot(int,const uint8_t*,const uint8_t*,const uint8_t*,float*);
typedef void (*quant_fn)(const float*,void*,int64_t);
int native_moe(int tokens,int topk,int experts,int h,int inter,
 const uint8_t *const *w1,const uint8_t *const *s1,
 const uint8_t *const *w3,const uint8_t *const *s3,
 const uint8_t *const *w2,const uint8_t *const *s2,
 const uint8_t *gpu_mask,const int64_t *ids,const float *router,
 const float *input,float *output,quant_fn quant) {
 if(tokens<0||topk<=0||experts<=0||h<=0||inter<=0||h%32||inter%32||
 !w1||!s1||!w3||!s3||!w2||!s2||!ids||!router||!input||!output||!quant) return -1;
 for(int e=0;e<experts;e++) {
   if(!w1[e]||!s1[e]||!w3[e]||!s3[e]||!w2[e]||!s2[e]) return -1;
   size_t scales=(size_t)h*inter/32;
   for(size_t i=0;i<scales;i++) if(s1[e][i]==255||s3[e][i]==255||s2[e][i]==255) return -2;
 }
 uint8_t *qh=malloc((size_t)h/32*34),*qi=malloc((size_t)inter/32*34);
 float *act=malloc((size_t)inter*sizeof(float));
 if(!qh||!qi||!act) {free(qh);free(qi);free(act);return -3;}
 for(int t=0;t<tokens;t++) {
   float *out=output+(size_t)t*h;
   memset(out,0,(size_t)h*sizeof(float));
   quant(input+(size_t)t*h,qh,h);
   for(int j=0;j<topk;j++) {
     int64_t e=ids[(size_t)t*topk+j];
     if(e<0||e>=experts||(gpu_mask&&gpu_mask[e])) continue;
     for(int row=0;row<inter;row++) {
       float gate,up;
       native_dot(h,w1[e]+(size_t)row*h/2,s1[e]+(size_t)row*h/32,qh,&gate);
       native_dot(h,w3[e]+(size_t)row*h/2,s3[e]+(size_t)row*h/32,qh,&up);
       act[row]=(gate/(1.0f+expf(-gate)))*up;
     }
     quant(act,qi,inter);
     float weight=router[(size_t)t*topk+j];
     for(int row=0;row<h;row++) {
       float down;
       native_dot(inter,w2[e]+(size_t)row*inter/2,s2[e]+(size_t)row*inter/32,qi,&down);
       out[row]+=down*weight;
     }
   }
 }
 free(qh);free(qi);free(act);return 0;
}
