import snowflake.snowpark.functions as F
import pandas as pd


def model(dbt, session):

    tuning_df = dbt.ref("hyperparameter_tuning_df")

    tune_results = tuning_df.select(
        F.col("hp_id"),
        (
            F.table_function("hyperparameter_tuning")(
                tuning_df["max_depth"],
                tuning_df["learning_rate"],
                tuning_df["n_estimators"],
                tuning_df["feature_vector"]
            ).over(partition_by="hp_id")
        )
    ).sort(F.col('RMSE').asc())
    best_param_id = tune_results.select("hp_id").collect()[0][0]
    params = (tuning_df.drop("feature_vector").filter(F.col("hp_id") == best_param_id).distinct())
    params = params.drop("hp_id").to_pandas().iloc[0, :].to_dict()

    return session.createDataFrame(pd.DataFrame(params, index=[0]))
