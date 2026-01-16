
**1. Create an IAM User**
  - Go to **IAM → Users → Create user**
  - Assign the following permission (attach policy directly):
    - **AWSLambda_FullAccess**

This permission allows you to deploy Lambda functions and push images to ECR.

**2. Generate AWS Access Keys**

After creating the IAM user:

1. Open the newly created user in the IAM console
2. Go to the **Security credentials** tab
3. Under **Access keys**, click **Create access key**
4. Save the generated:
   - **AWS Access Key ID**
   - **AWS Secret Access Key**

> These values are shown only once and must be stored securely.

**3. Create the Lambda Execution Role**
This role is used **by Lambda itself** when running your container.

Steps:

1. Go to **IAM → Roles → Create role**
2. **Trusted entity:** AWS Service  
3. **Use case:** Lambda
4. **Permissions:**  
   - `AWSLambdaBasicExecutionRole`  
     (allows Lambda to write logs to CloudWatch)
5. **Role name:**  
   ```
   lambda-execution-role
   ``` 