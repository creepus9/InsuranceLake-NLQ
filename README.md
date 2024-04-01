## Guidance for Natural Language Queries of over InsuranceLake on AWS

This soulution is based on [this repo](https://aws.amazon.com/solutions/guidance/natural-language-queries-of-relational-databases-on-aws/#) a demonstration of Generative AI, specifically, the use of Natural Language Query (NLQ) to
ask questions on top of Amazon Athena database. This solution uses Foundation Models via Amazon Bedrock. The demonstration's web-based application, running on Amazon ECS on AWS Fargate, uses a combination  
of [LangChain](https://python.langchain.com/docs/get_started/introduction.html), [Streamlit](https://streamlit.io/).
The application accepts natural language questions from end-users and returns natural
language answers, along with the associated SQL query and Pandas DataFrame-compatible result set.

#### NLQ Application Chatbot Preview

![NLQ Application Preview](./pics/nlq_animation.gif)

## Foundation Model Choice and Accuracy of NLQ

The selection of the Foundation Model (FM) for Natural Language Query (NLQ) plays a crucial role in the application's
ability to accurately translate natural language questions into natural language answers. Not all FMs are capable of
performing NLQ. In addition to model choice, NLQ accuracy also relies heavily on factors such as the quality of the
prompt, prompt template, labeled sample queries used for in-context learning (_aka few-shot prompting_), and the naming conventions used for the database schema, both tables and columns.

The NLQ Application was tested on a variety of FMs running on Amazon Bedrock and sepcificaly Amazon and Anthropic Claude families.

Amazon Titan Text G1 - Express, `amazon.titan-text-express-v1`, available through Amazon Bedrock, was tested. This model provided accurate responses to basic natural language queries using some model-specific prompt optimization. However, this model was not able to respond to more complex queries. Further prompt optimization could improve model accuracy.

Running Anthropic Claude (Calude 2.1 and Claude 3 - Haikou and Sonnet) generate the most accurate results.

## Sample Dataset

This solution uses an InsuranceLake athena based database, that contains a synthetic fictious dataset for policy and claims of a facticoius insurance company.
available on [GitHub](XXXXX). The database contains over XXXX and YYY...

Using this dataset, we can ask natural language questions, with varying levels of complexity:

- Simple
    - What is the average written premium for each line of business and employer size tier?
    - Who are the top 10 agents by total written premium?
    - What is the average claim amount for Commercial Auto policies, grouped by employer size tier and state?
    - What are the policies with the highest and lowest claim amounts for each line of business?
    - What are the top 5 industries with the highest total written premium for each line of business?
    - What are the top 5 agents with the highest written premium, and what are their distribution channels?
    - How many policies and what is the total earned premium for each combination of line of business and distribution channel?
    - What is the average claim amount and the average number of employees for each industry and sector?
    - How many policies and what is the total written premium for each combination of state and employer size tier, ordered by total written premium descending?
    - What is the average earned premium and the average claim amount for each combination of line of business and distribution channel, for policies that are in-force and have a claim amount greater than $50,000?
- Moderate
    - What is the average written premium for each line of business and employer size tier?
    - Who are the top 10 agents by total written premium?
    - What is the average claim amount for Commercial Auto policies, grouped by employer size tier and state?
    - What are the policies with the highest and lowest claim amounts for each line of business?
    - What are the top 5 industries with the highest total written premium for each line of business?
    - What are the top 5 agents with the highest written premium, and what are their distribution channels?
    - How many policies and what is the total earned premium for each combination of line of business and distribution channel?
    - What is the average claim amount and the average number of employees for each industry and sector?
    - How many policies and what is the total written premium for each combination of state and employer size tier, ordered by total written premium descending?
    - What is the average earned premium and the average claim amount for each combination of line of business and distribution channel, for policies that are in-force and have a claim amount greater than $50,000?
- Complex
    - For which combinations of line of business, distribution channel, and employer size tier is the premium retention ratio (earned premium / written premium) below 0.8?
    - What are the top 10 policies with the highest claim amounts for each combination of line of business, industry, and employer size tier?
    - For which combinations of line of business, industry, and territory is the policy retention rate (policies in-force / total policies) below 0.7?
    - What are the top 5 agents with the highest total written premium for each combination of line of business, industry, and territory, along with their average claim amount?
    - What are the policies with the highest and lowest earned premium for each combination of line of business, employer size tier, and state, along with their claim ratio (claim amount / earned premium)?
    - calculate the combined ratio for all data per each industry and state combination. return the data as a markdown table matrix
    - calculate the combined ratio for all data per each customer and state combination for the less profitable customers. return the data as a markdown table matrix
    - What are the top 5 territories with the highest total written premium, and what are the average claim amount and the average number of employees for each territory, ordered by total written premium descending?
    - How many policies, what is the total earned premium, and what is the average claim amount for each combination of line of business, distribution channel, and employer size tier, for policies that are new, have a claim amount greater than $100,000, and have a revenue greater than $10,000,000, ordered by total earned premium descending?
    - How many policies, what is the total written premium, and what is the average claim amount for each combination of industry, sector, and distribution channel, for policies that are expiring, have a claim amount between $50,000 and $200,000, and have a number of employees between 100 and 500, ordered by total written premium ascending?
    - How many policies, what is the total earned premium, and what is the average claim amount for each combination of state, city, and line of business, for policies that have a policy effective date in the year 2020, have a written premium greater than $50,000, and have a claim amount less than $20,000, ordered by total earned premium descending?
    - How many policies, what is the total written premium, and what is the average claim amount for each combination of agent name, line of business, and distribution channel, for policies that are new, have a policy effective date in the year 2021, have a written premium greater than $100,000, and have a claim amount greater than $75,000, ordered by total written premium descending?
- Unrelated to the Dataset
    - Give me a recipe for chocolate cake.
    - Who won the 2022 FIFA World Cup final?

Again, the ability of the NLQ Application to return an answer and return an accurate answer, is primarily dependent on
the choice of model. Not all models are capable of NLQ, while others will not return accurate answers. Optimizing the above prompts for specific models can help improve accuracy.

## Deployment Instructions (see details below)

1. 
2. Create the required secrets in AWS Secret Manager using the AWS CLI.
3. Deploy the `NlqMainStack` CloudFormation template. Please note, you will have needed to have used Amazon ECS at least one in your account, or the `AWSServiceRoleForECS` Service-Linked Role will not yet exist and the stack will fail.
   Check the `AWSServiceRoleForECS` Service-Linked Role before deploying the `NlqMainStack` stack. This role is auto-created the first time you create an ECS cluster in your account.
4. Build and push the `nlq-genai:2.0.0-bedrock` Docker image for use with Amazon Bedrock.
5. Create the Amazon RDS MoMA database tables and import the included sample data.
6. 
7. 
8. Deploy the `NlqEcsBedrockStack` CloudFormation template for use with Amazon Bedrock.

### Optional Step 1: Amazon SageMaker JumpStart Inference Instance

Not releveant for Amazon Bedrock

Ensure that you deployed the 2 InsuranceLake stacks - the infrastructure and the etl.

### Step 2: Create AWS Secret Manager Secrets

Not releveant for Amazon Bedrock and InsuranceLake

### Step 3: Deploy the Main NLQ Stack: Networking, Security, RDS Instance, and ECR Repository

Access to the ALB and RDS will be limited externally to your current IP address. You will need to update if
your IP address changes after deployment.

```sh
cd cloudformation/

aws cloudformation create-stack \
  --stack-name NlqMainStack \
  --template-body file://NlqMainStack.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters ParameterKey="MyIpAddress",ParameterValue=$(curl -s http://checkip.amazonaws.com/)/32
```

### Step 4: Build and Push the Docker Image to ECR

Build the Docker image(s) for the NLQ application, based on your choice of model options. You can build the Docker image(s) locally, in a CI/CD pipeline, using SageMaker Notebook environment, or AWS Cloud9. I prefer AWS Cloud9 for developing and testing the application and building the Docker images.

```sh
cd docker/

# Located in the output from the NlqMlStack CloudFormation template
# e.g. 111222333444.dkr.ecr.us-east-1.amazonaws.com/nlq-genai
ECS_REPOSITORY="<you_ecr_repository>"

aws ecr get-login-password --region us-east-1 | \
	docker login --username AWS --password-stdin $ECS_REPOSITORY
```

Amazon Bedrock:

```sh
TAG="2.0.0-bedrock"
docker build -f Dockerfile_Bedrock -t $ECS_REPOSITORY:$TAG .
docker push $ECS_REPOSITORY:$TAG
```


### Step 5: Configure Amazon Bedrock and InsuranceLake athena database access and credatials

Not relevant for Athena. Need to give credentials to the app access athena and bedrock.

### Step 6: Add NLQ Application to the MoMA Database

Access and permissions

### Optional Step 7: Deploy the Amazon SageMaker JumpStart Stack: Model and Endpoint

Not relevant

### Step 8: Deploy the ECS Service Stack: Task and Service

Amazon Bedrock:

```sh
aws cloudformation create-stack \
  --stack-name NlqEcsBedrockStack \
  --template-body file://NlqEcsBedrockStack.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

## Switching Foundation Models

### Alternate Amazon Bedrock Foundation Models

To switch from the solution's default Amazon Titan Text G1 - Express (`amazon.titan-text-express-v1`) Foundation Model,
you need to modify and rdeploy the `NlqEcsBedrockStack.yaml` CloudFormation template file. Additionally, you will need
to modify to the NLQ Application, `app_bedrock.py` Then, rebuild the Amazon ECR Docker Image using
the `Dockerfile_Bedrock`
Dockerfile and push the resulting image, e.g., `nlq-genai-2.0.1-bedrock`, to the Amazon ECR repository. Lastly, you will
need to
update the deployed ECS task and service, which are part of the `NlqEcsBedrockStack` CloudFormation stack.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
