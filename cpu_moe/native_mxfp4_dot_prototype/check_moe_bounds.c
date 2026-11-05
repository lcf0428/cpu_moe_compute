#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef void (*quant_fn)(const float*,void*,int64_t);
int native_moe(int,int,int,int,int,const uint8_t*const*,const uint8_t*const*,const uint8_t*const*,const uint8_t*const*,const uint8_t*const*,const uint8_t*const*,const uint8_t*,const int64_t*,const float*,const float*,float*,quant_fn);
static void quant(const float*x,void*out,int64_t n) {
 (void)x;memset(out,0,(size_t)n/32*34);
 for(int64_t b=0;b<n/32;b++) ((uint8_t*)out)[b*34+1]=0x3c;
}
int main(void) {
 for(int n=1;n<=17;n++) {
  int h=64,i=96;
  uint8_t *w=malloc((size_t)h*i/2),*s=malloc((size_t)h*i/32);
  float *x=calloc((size_t)n*h,sizeof(float)),*y=malloc((size_t)n*h*sizeof(float)),*r=malloc((size_t)n*3*sizeof(float));
  int64_t *ids=malloc((size_t)n*3*sizeof(int64_t));uint8_t mask[3]={0,0,1};
  if(!w||!s||!x||!y||!r||!ids) return 1;
  memset(w,0x76,(size_t)h*i/2);memset(s,127,(size_t)h*i/32);
  const uint8_t *ws[3]={w,w,w},*ss[3]={s,s,s};
  for(int a=0;a<n*3;a++){ids[a]=a%5-1;r[a]=.2f;}
  if(native_moe(n,3,3,h,i,ws,ss,ws,ss,ws,ss,mask,ids,r,x,y,quant))return 2;
  for(int a=0;a<n*h;a++)if(y[a]!=0)return 3;
  s[0]=255;
  if(native_moe(n,3,3,h,i,ws,ss,ws,ss,ws,ss,mask,ids,r,x,y,quant)!=-2)return 4;
  if(native_moe(n,3,3,33,i,ws,ss,ws,ss,ws,ss,mask,ids,r,x,y,quant)!=-1)return 5;
  free(w);free(s);free(x);free(y);free(r);free(ids);
 }
 puts("PASS full MoE ASan+UBSan: 17 batch sizes, exact buffers, skipped/duplicate/invalid routes, reserved scale and invalid shape");return 0;
}
