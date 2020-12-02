import boto3

s3client = boto3.client('s3', region_name='us-west-1')
objects = s3client.list_objects(Bucket='1000genomes')
# objects = s3client.list_objects(Bucket='epitome-data')
print(objects)
