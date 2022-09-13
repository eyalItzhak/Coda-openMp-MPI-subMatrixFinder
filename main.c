#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include "myProto.h"
#include <math.h>
#include <time.h>

typedef struct Matrix
{
  int ID;
  int dim;
  int **matix;
} Matrix;

Matrix *ReadMatrix(FILE *fp);
Matrix **ReadMatrix(FILE *fp, int *numberOfMatrixs);
Matrix **ReadFromFileInput(float *matchingValue, int *numberOfPictures, int *numberOfObjects, Matrix ***pictures);
Matrix **splitWork(Matrix **allPic, int num_picture, int numOfProcess, int Id_process);
float foo(int pic_num, int obj_num);
int findObjectInImage(int **picture, int **object, int dim_pic, int dim_obj, float threshold, int *pos_row, int *pos_col);

Matrix *ReadMatrix(FILE *fp)
{
  Matrix *matrix;
  int dim = 0;
  matrix = (Matrix *)malloc(sizeof(Matrix)); // matrix contain address of actual matix size...
  fscanf(fp, "%d", &(matrix->ID));
  fscanf(fp, "%d", &(matrix->dim));
  dim = matrix->dim;
  matrix->matix = (int **)malloc((dim) * (sizeof(int *))); // get dim address of address so every  single addres contain dim of ints....
  for (int i = 0; i < dim; i++)
  {
    matrix->matix[i] = (int *)malloc(dim * sizeof(int)); // single address contain dim of ints...
    for (int j = 0; j < dim; j++)
    {
      fscanf(fp, "%d", &(matrix->matix[i][j]));
    }
  }
  return matrix;
}

Matrix **ReadMatrix(FILE *fp, int *numberOfMatrixs)
{
  Matrix **matrixs = NULL;
  Matrix *matrix;
  fscanf(fp, "%d", numberOfMatrixs); // Read number of pictures
  printf("Read matching value %d\n", *numberOfMatrixs);
  if (*numberOfMatrixs <= 0)
  {
    printf("No pictures to search in\n");
    exit(1);
  }
  for (int i = 0; i < *numberOfMatrixs; i++)
  {
    matrix = ReadMatrix(fp);
    matrixs = (Matrix **)realloc(matrixs, (sizeof(Matrix) * (i + 1)));
    matrixs[i] = matrix;
  }
  return matrixs;
}

Matrix **ReadFromFileInput(float *matchingValue,
                           int *numberOfPictures, int *numberOfObjects, Matrix ***pictures)
{
  FILE *fp;
  Matrix **objects = NULL;
  // Open file for reading points
  if ((fp = fopen("/home/linuxu/Desktop/temp/Ex5/Input.txt", "r")) == 0)
  {
    printf("cannot open file or reading\n");
    exit(1);
  }
  printf("open file  for reading\n");
  fscanf(fp, "%f", matchingValue); // Read matching value
  printf("Read matching value %f\n", *matchingValue);
  *pictures = ReadMatrix(fp, numberOfPictures);
  objects = ReadMatrix(fp, numberOfObjects);

  fclose(fp);
  printf("done reading\n");
  return objects;
}

float foo(int pic_num, int obj_num)
{

  double myRes = double(double(pic_num - obj_num) / double(pic_num));

  double res = fabs(myRes);
  return (float)res;
}

int findObjectInImage(int **picture, int **object, int dim_pic, int dim_obj, float threshold, int *pos_row, int *pos_col)
{
  int itrations = dim_pic - dim_obj;
  float sum;
  int object_elment, picture_element;
  int asn = 0;
  for (int offset_row = 0; offset_row < itrations; offset_row++)
  {
    for (int offset_col = 0; offset_col < itrations; offset_col++)
    {
      sum = 0;
      for (int i = 0; i < dim_obj; i++)
      {
        for (int j = 0; j < dim_obj; j++)
        {
          object_elment = object[i][j];
          picture_element = picture[i + offset_row][j + offset_col];
          //  printf("from gpu \n");
          sum = sum + foo(picture_element, object_elment);
          if (sum > threshold)
          {
            i = dim_obj;
            j = dim_obj;
          }
        }
      }
      if (sum <= threshold)
      {
        asn = 1;
        *pos_row = offset_row;
        *pos_col = offset_col;
        offset_col = itrations;
        offset_row = itrations;
      }
    }
  }
  return asn;
}

int main(int argc, char *argv[])
{
  clock_t begin, end;
  float M_Val = 0;
  int src = 0;
  int des = 0;
  int size, rank, i, j, index;
  int tag = 0;
  int mySize;
  int num_picture = 0, num_objects = 0;

  struct Matrix **pictures = NULL;
  struct Matrix **objects = NULL;

  struct Matrix **subPics = NULL;
  struct Matrix **myPics = NULL;
  int newSize;

  MPI_Status status;
  int resultOMP = 0, resultCUDA = 0, resultTotal = 0;
  for (i = 0; i < argc; i++)
  {
    printf("argc: %s\n", argv[i]);
  }
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2)
  { // configer
    printf("Run the example with 2 processes only\n");
    MPI_Abort(MPI_COMM_WORLD, __LINE__);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
  {
    begin = clock();
    objects = ReadFromFileInput(&M_Val, &num_picture, &num_objects, &pictures);
  }
  MPI_Bcast(&M_Val, 1, MPI_FLOAT, src, MPI_COMM_WORLD);
  MPI_Bcast(&num_picture, 1, MPI_FLOAT, src, MPI_COMM_WORLD);
  MPI_Bcast(&num_objects, 1, MPI_FLOAT, src, MPI_COMM_WORLD);

  //   printf("done MPI_Bcast get flat =>%f ,pic=>%d\n from rank__ %d\n",M_Val,num_picture,rank);
  if (rank == 0)
  {
    for (i = 1; i < size; i++)
    { // sent to evert other process

      des = i; // the process we sent to
      for (j = 0; j < num_picture; j++)
      {                                                                    // for every pic
        MPI_Send(&pictures[j]->ID, 1, MPI_INT, des, tag, MPI_COMM_WORLD);  // sent is id
        MPI_Send(&pictures[j]->dim, 1, MPI_INT, des, tag, MPI_COMM_WORLD); // sent is dim
        for (index = 0; index < pictures[j]->dim; index++)
        {                                                                                           // sent is matrix
          MPI_Send(pictures[j]->matix[index], pictures[j]->dim, MPI_INT, des, tag, MPI_COMM_WORLD); // sent to process form pic j=> all element of a row (matrix)
        }
      }

      for (j = 0; j < num_objects; j++)
      {                                                                   // for every pic
        MPI_Send(&objects[j]->ID, 1, MPI_INT, des, tag, MPI_COMM_WORLD);  // sent is id
        MPI_Send(&objects[j]->dim, 1, MPI_INT, des, tag, MPI_COMM_WORLD); // sent is dim
        for (index = 0; index < objects[j]->dim; index++)
        {                                                                                         // sent is matrix
          MPI_Send(objects[j]->matix[index], objects[j]->dim, MPI_INT, des, tag, MPI_COMM_WORLD); // sent to process form pic j=> all element of a row (matrix)
        }
      }
    }
  }
  else
  {
    for (i = 0; i < num_picture; i++)
    {                                                                             // get pics
      Matrix *temp_matrix = (Matrix *)malloc(sizeof(Matrix));                     // expect to get matrix
      MPI_Recv(&temp_matrix->ID, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);  // get id
      MPI_Recv(&temp_matrix->dim, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status); // get dim
      //  printf( "id=>%d\n,dim=>%d\n from rank__ %d\n",temp_matrix->ID,temp_matrix->dim,rank);
      temp_matrix->matix = (int **)malloc((temp_matrix->dim) * (sizeof(int *)));
      for (j = 0; j < temp_matrix->dim; j++)
      {                                                                                                // from dim =>expected dim of  arrays (marix)
        temp_matrix->matix[j] = (int *)malloc(temp_matrix->dim * sizeof(int));                         // new row of elements
        MPI_Recv(temp_matrix->matix[j], temp_matrix->dim, MPI_INT, src, tag, MPI_COMM_WORLD, &status); // get to process form pic j=> all element of a row (matrix)
      }
      pictures = (Matrix **)realloc(pictures, (sizeof(Matrix) * (i + 1))); // set arrys of pic to the new size
      pictures[i] = temp_matrix;                                           // put the value .
    }

    for (i = 0; i < num_objects; i++)
    {                                                                             // get objects
      Matrix *temp_matrix = (Matrix *)malloc(sizeof(Matrix));                     // expect to get matrix
      MPI_Recv(&temp_matrix->ID, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status);  // get id
      MPI_Recv(&temp_matrix->dim, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &status); // get dim
      // printf( "id=>%d\n,dim=>%d\n from rank__ %d\n",temp_matrix->ID,temp_matrix->dim,rank);
      temp_matrix->matix = (int **)malloc((temp_matrix->dim) * (sizeof(int *)));
      for (j = 0; j < temp_matrix->dim; j++)
      {                                                                                                // from dim =>expected dim of  arrays (marix)
        temp_matrix->matix[j] = (int *)malloc(temp_matrix->dim * sizeof(int));                         // new row of elements
        MPI_Recv(temp_matrix->matix[j], temp_matrix->dim, MPI_INT, src, tag, MPI_COMM_WORLD, &status); // get to process form pic j=> all element of a row (matrix)
      }
      objects = (Matrix **)realloc(objects, (sizeof(Matrix) * (i + 1))); // set arrys of pic to the new size
      objects[i] = temp_matrix;                                          // put the value .
    }
  }
  int *found = (int *)malloc(sizeof(int) * num_picture);
  int *pos_x = (int *)malloc(sizeof(int) * num_picture);
  int *pos_y = (int *)malloc(sizeof(int) * num_picture);
  for (i = 0; i < num_picture; i++)
  {
    found[i] = -1;
  }

#pragma omp parallel private(i) shared(found, pos_x, pos_y)
  {
#pragma omp for schedule(static)
    for (i = 0; i < num_picture; i++)
    {
      int res;
      if (i % size == rank)
      {
        for (int j = 0; j < num_objects; j++)
        {
          int object_elements = (objects[j]->dim);
          int pic_elements = (pictures[i]->dim);
          int itrations = pic_elements - object_elements;
          int pos_row, pos_col;
          res = findObjectInImage((pictures[i]->matix), (objects[j]->matix), pictures[i]->dim, objects[j]->dim, M_Val, &pos_row, &pos_col);
          if (res == 1)
          {
            found[i] = j;
            pos_x[i] = pos_row;
            pos_y[i] = pos_col;
            j = num_objects;
          }
        }
      }
    }
  }

  if (rank == 0)
  {
    int *found_recv = (int *)malloc(sizeof(int) * num_picture);
    int *pos_x_recv = (int *)malloc(sizeof(int) * num_picture);
    int *pos_y_recv = (int *)malloc(sizeof(int) * num_picture);

    MPI_Recv(found_recv, num_picture, MPI_INT, des, tag, MPI_COMM_WORLD, &status); // get id

    MPI_Recv(pos_x_recv, num_picture, MPI_INT, des, tag, MPI_COMM_WORLD, &status); // get dim

    MPI_Recv(pos_y_recv, num_picture, MPI_INT, des, tag, MPI_COMM_WORLD, &status); //
    FILE *fp;

    fp = fopen("/home/linuxu/Desktop/temp/Ex5/output.txt", "w"); // abosolt path

    for (i = 0; i < num_picture; i++)
    {
      if (found_recv[i] != -1)
      {
        printf("found object  %d in pic %d pos {%d,%d} \n", found_recv[i], i, pos_x_recv[i], pos_y_recv[i]);
        fprintf(fp, "found object  %d in pic %d pos {%d,%d} \n", found_recv[i], i, pos_x_recv[i], pos_y_recv[i]);
      }
      else if (found[i] != -1)
      {
        printf("found object  %d in pic %d pos {%d,%d} \n", found[i], i, pos_x[i], pos_y[i]);
        fprintf(fp, "found object  %d in pic %d pos {%d,%d} \n", found[i], i, pos_x[i], pos_y[i]);
      }
      else
      {
        printf("not found object in pic %d  \n", i);
        fprintf(fp, "not found object in pic %d  \n", i);
      }
    }
    fclose(fp);
    free(found_recv);
    free(pos_x_recv);
    free(pos_y_recv);
  }
  else
  {
    // printf("sent data  =>info %d \n ",found[0]);
    MPI_Send(found, num_picture, MPI_INT, src, tag, MPI_COMM_WORLD);
    MPI_Send(pos_x, num_picture, MPI_INT, src, tag, MPI_COMM_WORLD);
    MPI_Send(pos_y, num_picture, MPI_INT, src, tag, MPI_COMM_WORLD);
  }
  free(found);
  free(pos_x);
  free(pos_y);

  for (int i = 0; i < num_objects; i++)
  {
    for (int j = 0; i < objects[i]->ID; j++)
    {
      free(objects[i]->matix[j]);
      free(objects[i]->matix);
      free(objects[i]);
    }
  }

  for (int i = 0; i < num_picture; i++)
  {
    for (int j = 0; i < pictures[i]->ID; j++)
    {
      free(pictures[i]->matix[j]);
      free(pictures[i]->matix);
      free(pictures[i]);
    }
  }
  if (rank == 0)
  {
    end = clock();
    double time_spen = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time %lf sec \n", time_spen);
  }

  MPI_Finalize();
}
