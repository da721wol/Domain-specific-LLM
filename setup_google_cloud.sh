export GCP_ZONE="us-central1-a"
export GCP_TPU_NAME="mytpucluster"
export GCP_PROJECT_ID="masterarbeit-verwaltung"

# >>>>> Run this part to add a disk to your TPU VM
export GCP_DISK_NAME="datadisk"
export GCP_DISK_SIZE_GB=1200
export GCP_DISK_TYPE=pd-standard

#gcloud compute disks create $GCP_DISK_NAME \
#    --project=$GCP_PROJECT \
#    --type=$GCP_DISK_TYPE \
#    --size="${GCP_DISK_SIZE_GB}GB" \
#    --zone=$GCP_ZONE \
#    --project=$GCP_PROJECT_ID


# Create the TPU VM
while true; do
  gcloud compute tpus tpu-vm create $GCP_TPU_NAME \
      --zone $GCP_ZONE \
      --accelerator-type v3-8 \
      --version v2-alpha \
      --project=$GCP_PROJECT_ID
      #--data-disk source="projects/${GCP_PROJECT}/zones/${GCP_ZONE}/disks/${GCP_DISK_NAME}"
  RETURN_CODE=$?
  if [ $RETURN_CODE -eq 0 ]; then
      echo "Command executed successfully"
      break
  fi
  sleep 600
done


(cd .. && gcloud compute tpus tpu-vm scp --recurse ./t5-flax-gcp $GCP_TPU_NAME:~/t5 --zone $GCP_ZONE --project=$GCP_PROJECT_ID)
gcloud compute tpus tpu-vm ssh $GCP_TPU_NAME --zone $GCP_ZONE --project=$GCP_PROJECT_ID