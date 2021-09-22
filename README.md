# cust-churn-dmm

ADD A NEW COMENT

code to push cust churn prediction/ground truth data to s3 and then DMM - September 22!!!!

## Assets

CreateHoldout.ipynb - Notebook used for testing. Keeping this for easier interpretability and testing

upload_to_s3.py - Helper script with functions for setting up and executing upload of files to specified s3 bucket (thanks to Colin Goyette!)

model-inference.py - Primary script that is scheduled to run daily to provide updated files to S3 bucket for DMM to pull from



