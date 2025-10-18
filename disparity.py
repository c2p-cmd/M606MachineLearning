# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "imblearn==0.0",
#     "openai==2.3.0",
#     "pandas==2.3.3",
#     "plotly==6.3.1",
#     "scikit-learn==1.7.2",
#     "nbconvert",
# ]
# ///

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    return mo, pd, px


@app.cell
def _(mo):
    mo.md("""# Understanding classification bias""")
    return


@app.cell
def _(pd):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Mahmoudreza/teaching/refs/heads/main/datasets/adult.csv"
    )
    df
    return (df,)


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    from imblearn.over_sampling import SMOTE
    return (
        LabelEncoder,
        MLPClassifier,
        OrdinalEncoder,
        RandomForestClassifier,
        SMOTE,
        SVC,
        train_test_split,
    )


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    df.describe(include="all")
    return


@app.cell
def _(df, mo, px):
    _data = (
        df.target.value_counts(normalize=True).rename("Proportion").to_frame()
        * 100
    )
    _fig = px.pie(_data, names=_data.index, values="Proportion")

    mo.vstack([mo.md("## Original Balance"), mo.ui.plotly(_fig)])
    return


@app.cell
def _(LabelEncoder, OrdinalEncoder, SMOTE, df, pd, train_test_split):
    marital_status_encoder = OrdinalEncoder()
    gender_encoder = OrdinalEncoder()
    label_encoder = LabelEncoder()
    smote = SMOTE(random_state=42)


    def create_dataframe(df):
        new_df = df.copy()[["sex", "marital-status", "target"]]
        new_df_train, new_df_test = train_test_split(
            new_df, test_size=0.2, random_state=42, stratify=new_df["target"]
        )

        new_df_train = new_df_train.reset_index(drop=True)
        new_df_test = new_df_test.reset_index(drop=True)

        new_df_train["sex"] = gender_encoder.fit_transform(new_df_train[["sex"]])
        new_df_test["sex"] = gender_encoder.transform(new_df_test[["sex"]])

        new_df_train["target"] = label_encoder.fit_transform(
            new_df_train["target"]
        )
        new_df_test["target"] = label_encoder.transform(new_df_test["target"])

        new_df_train["marital-status"] = marital_status_encoder.fit_transform(
            new_df_train[["marital-status"]]
        )
        new_df_test["marital-status"] = marital_status_encoder.transform(
            new_df_test[["marital-status"]]
        )
        X_res, y_res = smote.fit_resample(
            new_df_train.drop(columns=["target"]), new_df_train["target"]
        )
        new_df_train = pd.concat([X_res, y_res], axis=1)
        new_df_train = new_df_train.sample(frac=1, random_state=42).reset_index(
            drop=True
        )
        return new_df_train, new_df_test


    df_train, df_test = create_dataframe(df)
    return df_test, df_train, gender_encoder, label_encoder


@app.cell
def _(df_test, df_train, mo):
    mo.md(
        f"""
    * Train shape: `{df_train.shape}`
    * Test Shape: `{df_test.shape}`
    """
    )
    return


@app.cell
def _(df_train, mo):
    mo.ui.table(df_train, label="Training DataFrame")
    return


@app.cell
def _(mo):
    mo.md("""## EDA of df_train""")
    return


@app.cell
def _(df_train, mo, px):
    _fig = px.pie(
        df_train,
        names="target",
        title="Target Distribution in Training Set",
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(df_train, mo, px):
    _fig = px.scatter(
        data_frame=df_train,
        x="marital-status",
        y="sex",
        color="target",
        title="Marital Status vs Gender colored by Target",
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(MLPClassifier, RandomForestClassifier, SVC):
    # declare 3 classifiers
    clf1 = RandomForestClassifier(random_state=42, criterion='log_loss')
    clf2 = SVC(random_state=42)
    clf3 = MLPClassifier(
        random_state=42,
        max_iter=500,
        hidden_layer_sizes=(50, 25),
    )
    return clf1, clf2, clf3


@app.cell
def _(clf1, clf2, clf3, mo):
    mo.vstack(
        [
            mo.md("## Defining 3 classifiers"),
            mo.hstack([clf1, clf2, clf3]),
        ]
    )
    return


@app.cell
def _(clf1, clf2, clf3, df_test, df_train):
    def train_and_evaluate(clf, df_train, df_test):
        X_train = df_train.drop(columns=["target"])
        y_train = df_train["target"]
        X_test = df_test.drop(columns=["target"])
        y_test = df_test["target"]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred


    reports = {}
    cms = {}
    for i, clf in enumerate([clf1, clf2, clf3], start=1):
        reports[f"Classifier {i}"] = train_and_evaluate(clf, df_train, df_test)
    return (reports,)


@app.cell
def _(df_test, gender_encoder, label_encoder, mo, pd):
    def disparity_table(y_pred):
        df_temp = df_test.copy()
        df_temp["Prediction"] = label_encoder.inverse_transform(y_pred)
        disparity = (
            df_temp.groupby(["sex", "Prediction"]).size().unstack(fill_value=0)
        )
        disparity.index = gender_encoder.inverse_transform(
            disparity.index.to_frame()
        ).flatten()
        disparity["Total"] = disparity.sum(axis=1)
        disparity["Proportion_>50K"] = (
            disparity[" >50K"] / disparity["Total"]
        ) * 100
        return mo.ui.table(disparity, label="## Disparity Impact")


    def disparity_mistreatment(y_pred):
        df_temp = df_test.copy()
        df_temp["Prediction"] = label_encoder.inverse_transform(y_pred)
        df_temp["Actual"] = label_encoder.inverse_transform(df_temp["target"])
        df_temp["Gender"] = gender_encoder.inverse_transform(
            df_temp[["sex"]]
        ).flatten()
        accuracy = (
            df_temp.groupby("Gender")
            .apply(lambda x: (x["Prediction"] == x["Actual"]).mean())
            .reset_index()
        )
        accuracy = pd.DataFrame(accuracy)
        accuracy.columns = ["Gender", "Accuracy"]
        return mo.ui.table(
            accuracy,
            label="## Disparity Mistreatment (Accuracy)",
        )


    def disparity_treatment(y_pred):
        df_temp = df_test.copy()
        df_temp["Prediction"] = label_encoder.inverse_transform(y_pred)
        df_temp["Actual"] = label_encoder.inverse_transform(df_temp["target"])
        df_temp["Gender"] = gender_encoder.inverse_transform(
            df_temp[["sex"]]
        ).flatten()
        precision = (
            df_temp.groupby("Gender")
            .apply(lambda x: (x["Prediction"] != x["Actual"]).mean())
            .reset_index()
        )
        precision = pd.DataFrame(precision)
        precision.columns = ["Gender", "Error Rate"]
        return mo.ui.table(precision, label="## Disparity Treatment")
    return disparity_mistreatment, disparity_table, disparity_treatment


@app.cell
def _(
    disparity_mistreatment,
    disparity_table,
    disparity_treatment,
    mo,
    reports,
):
    mo.vstack(
        [
            mo.md("## Predictions by **Random Forest**"),
            disparity_table(reports["Classifier 1"]),
            disparity_mistreatment(reports["Classifier 1"]),
            disparity_treatment(reports["Classifier 1"]),
        ]
    )
    return


@app.cell
def _(
    disparity_mistreatment,
    disparity_table,
    disparity_treatment,
    mo,
    reports,
):
    mo.vstack(
        [
            mo.md("## Predictions by **SVM**"),
            disparity_table(reports["Classifier 2"]),
            disparity_mistreatment(reports["Classifier 2"]),
            disparity_treatment(reports["Classifier 2"]),
        ]
    )
    return


@app.cell
def _(
    disparity_mistreatment,
    disparity_table,
    disparity_treatment,
    mo,
    reports,
):
    mo.vstack(
        [
            mo.md("## Predictions by **MLP (Neural Network)**"),
            disparity_table(reports["Classifier 3"]),
            disparity_mistreatment(reports["Classifier 3"]),
            disparity_treatment(reports["Classifier 3"]),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
