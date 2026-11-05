#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
int native_dot(int,const uint8_t*,const uint8_t*,const uint8_t*,float*);
int main(void) {
    int sizes[]={32,64,96,128,160,256,512,2048,4096,8192};
    for(int t=0;t<10;t++) for(int j=0;j<100;j++) {
        int k=sizes[t],nb=k/32;
        uint8_t *w=malloc(k/2),*s=malloc(nb),*q=malloc(nb*34);
        if(!w||!s||!q) return 1;
        for(int i=0;i<k/2;i++) w[i]=(uint8_t)(i*17+j);
        for(int b=0;b<nb;b++) {
            s[b]=(uint8_t)(110+b%20);
            q[34*b]=0; q[34*b+1]=0x3c; /* fp16 1 */
            for(int i=2;i<34;i++) q[34*b+i]=(uint8_t)(i+j);
        }
        float out;
        if(native_dot(k,w,s,q,&out)) return 2;
        free(w);free(s);free(q);
    }
    puts("PASS ASan+UBSan: 1000 exact-size allocations, odd/even block counts");
    return 0;
}
