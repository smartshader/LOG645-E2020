__global__ void plus100Kernel(int *input, int* output)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < 10000)
	{
		output[i] = input[i] + 100;
	}
}
void plus100()
{
	int *d_input = 0;
	int *d_output = 0;
	
	cudaMalloc((void**)&d_input, 10000 * sizeof(int));
	cudaMalloc((void**)&d_output, 10000 * sizeof(int));
	
	srand(time(NULL));
	
	int* matrice = (int*)malloc(sizeof(int) * 10000);
	for (int i = 0; i < 10000; i++)
	{
		matrice[i] = rand() % 100;
	}
	
	// Copier vers le dispositif
	cudaMemcpy(d_input, matrice, 10000 * sizeof(int), cudaMemcpyHostToDevice);
	
	// Appeler le kernel avec 256 blocs
	plus100Kernel<<<256, 1>>>(d_input, d_output);
	
	// Attendre que le kernel ait fini, puis copier vers l'hôte
	cudaDeviceSynchronize();
	cudaMemcpy(matrice, d_output, 10000 * sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 10000; i++)
	{
		printf("%d\n", matrice[i]);
	}
}