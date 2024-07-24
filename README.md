# classification-tflite

## Testing

We test this when you cut a PR by running `scripts/test.sh`. This in turn just runs the model and then runs `pytest` which will run everything in `tests/`. Our testing here isn't overly complex and we have no code coverage or anything so please keep up good hygiene and if you make a change to `training.py` that would requires a test (i.e. you've made a change and you had to manually test something different), add that test condition somewhere to `tests/`

## Workflows

### Pull Request

When you submit a pull request a workflow will run using our [common workflows](https://github.com/viam-modules/common-workflows/) that will lint check your code, build it in the docker image we us in production (tensorflow/tensorflow:2.11.1-gpu) and run the test file you specify.

The default test files is `scripts/test.sh`. If this changes you will need to update `.github/workflows/pull_request.yaml` so that it's

```
jobs:
  build:
    uses: viam-modules/common-workflows/.github/workflows/lint_and_test.yaml
    with:
      test_script_name: NEW_TEST_FILE_NAME
```

### Main

Upon merging to `main` a workflow will automatically update the module in `viam-dev` allowing for people to use your latest changes. The configs you can (but shouldn't!) play with are:
1. framework -- DO NOT CHANGE THIS! This is a tflite script and will always be (see: repo name)
2. script_name -- This is what the name will be in the registry. If you change this, it will make a new training script in the registry. Be aware
3. model_type -- DO NOT CHANGE THIS!
