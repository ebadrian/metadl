# Docker image for MetaDL competition

The Docker image used for MetaDL challenge is created with `Dockerfile`
in this directory.

Here is the [link](https://hub.docker.com/r/ebadrian/metadl/) to Docker Hub.

## Build Docker image for GPU/CPU
For GPU, use
```
docker build -t ebadrian/metadl:gpu .
```
For CPU, use
```
docker build -t ebadrian/metadl:cpu -f Dockerfile.cpu .
```
If you have push access, you can do
```
docker push ebadrian/metadl:gpu
```
or
```
docker push ebadrian/metadl:cpu
```

For simplicity, two scripts are written to automate the build
```
./build.sh
```
and the release
```
./release.sh
```

## Creating a compute worker for the CodaLab competition 
The `worker_setup.sh` is a shell script created to install some modules and packages in an Azure VM. More specifically the VM should have Unbuntu 18.04 (x86_64). Once created, you can clone the GitHub repository with the following command : 

```bash
git clone https://github.com/ebadrian/metadl.git
```

Then you can run the `worker_setup.sh` script to install all the required softwares/packages to create a compute worker for this competition.

```bash
cd metadl/docker/
bash ./worker_setup.sh
```

The process may take several minutes and user input is required to accept the Here are the softwares/packages installed during the process: 

- **Gcc** compiler and associated packages
- Nvidia **cuda** 
- **Docker**
- **Nvidia docker**

**Note:** If you want to use such a worker for another CodaLab [workers queue](https://github.com/codalab/codalab-competitions/wiki/User_Using-your-own-compute-workers#hooking-up-a-compute-worker-to-a-queue),
you must change the environment variable **BROKER_URL** with the corresponding 
CodaLab queue identifier. More detailled informations about this can be found [in the dedicated wiki](https://github.com/codalab/codalab-competitions/wiki/User_Using-your-own-compute-workers#step-2-configuration-file).