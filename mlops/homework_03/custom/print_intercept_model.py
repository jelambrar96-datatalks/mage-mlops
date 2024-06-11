if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(data, *args, **kwargs):
    __,__,__,model=data
    print("intercept", model.intercept_)
    return {}


@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
