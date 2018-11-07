#ifndef _AHPG_KERNEL_NOT_INCLUDED_
#define _AHPG_KERNEL_NOT_INCLUDED_

static __global__
void acquisitonKernel(float* data, float* min_max, float* result, int data_size, int result_size){
	// //The row is the number of the alternative
	// int row = blockIdx.x*size+threadIdx.x;
	// //The col is the number of the criteria
	// int col = blockIdx.y*size+threadIdx.y;
	// int value_alt1_int, value_alt2_int, t;
	// float value_alt1_float, value_alt2_float;
	// char* sub,*sub2;
	// // int k=0;
	// // for(int i=0; data[i]!='\0'; i++) {
	// //      if(i==index[k]) {printf("|"); k++;}
	// //      printf("%c",data[i]);
	// // }
	// // printf("ALT SIZE: %d , CRIT SIZE %d\n",size, sizeCrit);
	// // if(row==0 && col <sizeCrit) { //the thread can do the work
	// if(row<size && col <sizeCrit) {                 //the thread can do the work
	//      int indexRead = row*sizeCrit+col;
	//      // printf("NEW THREAD ROW %d COL %d SIZE %d SIZECRIT %d\n",row,col,size,sizeCrit);
	//      value_alt1_int=0;
	//      value_alt1_float=0.0f;
	//      t=0;
	//      // printf("%d # %d # %d # %d\n",indexRead, indexRead+1,index[indexRead],index[indexRead+1]);
	//      sub=copyStr(data,index[indexRead],index[indexRead+1]);
	//      if(types[row*sizeCrit+col]==0 || types[row*sizeCrit+col]==2) {
	//              value_alt1_int=char_to_int(sub);
	//              // printf("CONVERTED INT %d\n",value_alt1_int);
	//      }else if(types[row*sizeCrit+col]==1) {
	//              value_alt1_float=char_to_float(sub);
	//              // printf("CONVERTED FLOAT %f\n",value_alt1_float);
	//      }
	//      for(int alt=0; alt<size; alt++) {
	//              sub2=copyStr(data,index[alt*sizeCrit+col],index[alt*sizeCrit+(col+1)]);
	//              // printf("ALTERNATIVE %d - %s # %s\n",alt,sub,sub2);
	//              value_alt2_int=0;
	//              value_alt2_float=0.0f;
	//              //alt*sizeCrit+col will jump over the alternatives to get the same coleria value.
	//              if(types[alt*sizeCrit+col]==0) {
	//                      t=0;
	//                      value_alt2_int=char_to_int(sub2);
	//              }else if(types[alt*sizeCrit+col]==1) {
	//                      t=1;
	//                      value_alt2_float=char_to_float(sub2);
	//              }else if(types[alt*sizeCrit+col]==2) {
	//                      t=2;
	//                      value_alt2_int=char_to_int(sub2);
	//              }
	//              // printf("DIVIDED BY %f\n",max_min[col]);
	//              // printf("SIZE: %d\n",size);
	//              int indexWrite = row*size*sizeCrit+col*size+alt;
	//              // int indexWrite = row*size*size/2+alt;
	//              // printf("WERE I WRITE %d\n",row*size*sizeCrit+col*size+alt);
	//              //Its used row*size*size/2 to jump correctly in the vector and set the values
	//              if(t==0) {
	//                      value_alt1_int==value_alt2_int ? cmp[indexWrite]=1 : cmp[indexWrite] = (value_alt1_int-value_alt2_int) / (float) max_min[col];
	//                      // printf("Write in T0 %f\n",cmp[indexWrite]);
	//              }else if(t==1) {
	//                      // if(value_alt1_float!=value_alt2_float) printf("DIF %f %f\n",value_alt1_float,value_alt2_float);
	//                      value_alt1_float==value_alt2_float ? cmp[indexWrite]=1 : cmp[indexWrite] = (value_alt1_float - value_alt2_float) / (float) max_min[col];
	//                      // printf("Write in T1 %f\n",cmp[indexWrite]);
	//              }
	//              else if(t==2) {
	//                      // printf("ALTERNATIVE %d - %s # %s\n",alt,sub,sub2);
	//                      // printf("BOOL %d %d\n",value_alt1_int,value_alt2_int);
	//                      if(value_alt1_int==value_alt2_int) cmp[indexWrite]=1;
	//                      else if(value_alt1_int==1) cmp[indexWrite]=9;
	//                      else if(value_alt1_int==0) cmp[indexWrite]=1/9.0f;
	//                      // printf("Write in T2 %f\n", cmp[indexWrite]);
	//              }else{
	//                      printf("UNESPECTED VALUE FOR T\n");
	//              }
	//      }
	// }
}

#endif
