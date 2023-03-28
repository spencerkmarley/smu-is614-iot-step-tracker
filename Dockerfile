FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN pip3 install -r requirements.txt #--target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY src ./src
WORKDIR ${LAMBDA_TASK_ROOT}

#ENTRYPOINT ["python", "-m", "src.inference.handler"]
##include the default file to be executed.
CMD ["src.inference.lambda_handler"]