import streamlit as st
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score, classification_report, precision_recall_curve, roc_curve, roc_auc_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

st.title('Data Pre-Processing Operator')
# st.divider()

st.header('File Selection')
uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    # st.write(string_data)
    max_num = max(string_data.count(','), string_data.count('\t'), string_data.count(';'), string_data.count('|'))
    if string_data.count('|') == max_num:
        sep = '|'
    elif string_data.count('\t') == max_num:
        sep = '\t'
    elif string_data.count(';')  == max_num:
        sep = ';'
    else:
        sep = ','

    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file, encoding='cp949', sep=sep)

    # st.divider()
    st.header('DataFrame')
    with st.expander("See DataFrame"):
        st.dataframe(df, use_container_width=True)

    # st.divider()
    # st.write("")
    # st.write("")

    st.header('Before : Detail of the Data')
    tab1, tab2, tab3 = st.tabs(["Information", "Description", "Correlation"])

    with tab1:
        st.header("Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        st.text(df_info)

    with tab2:
        st.header("Description")
        description_options = st.selectbox(
        'Select Type',
        ['Numeric', 'Category', 'All'])
        if description_options == 'Numeric':
            try:
                df_describe = df.describe(exclude='object')
            except:
                st.write("No Numerical Data")
        elif description_options == 'Category':
            try:
                df_describe = df.describe(include='object')
            except:
                st.write("No Categorical Data")
        else:
            df_describe = df.describe(include='all')
        try:
            st.write(df_describe)
        except:
            st.write('')

    with tab3:
        st.header("Correlation")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), ax=ax, cmap='Reds', annot=True, annot_kws={"size": 7})
        st.write(fig)

    # st.write("")
    # st.write("")
    # st.divider()

    with st.expander("Operate Data Preprocessing"):

        st.header('Null Operation')
        df_columns_origin = df.columns
        df_columns = df.columns
        # st.write(df_columns)
        df_new = df.copy()

        null_operation = st.selectbox(
            'Do You Need Null Operation?',
            ('No, I Do Not', 'Yes, I Do'))

        if null_operation == 'Yes, I Do':
            with st.sidebar:

                # st.divider()
                st.subheader("Drop Null")
                drop_null_options = st.multiselect(
                    'What columns drop nulls?',
                    df_columns)
                df_columns = df_columns.difference(drop_null_options, sort=False)

                # st.divider()
                st.subheader("Fill Mean Value into Null")
                fill_mean_options = st.multiselect(
                    'What columns fill mean?',
                    df_columns)
                df_columns = df_columns.difference(fill_mean_options, sort=False)

                # st.divider()
                st.subheader("Fill Median Value into Null")
                fill_median_options = st.multiselect(
                    'What columns fill median?',
                    df_columns)
                df_columns = df_columns.difference(fill_median_options, sort=False)


                # st.divider()
                st.subheader("Fill 0 into Null")
                fill_zero_options = st.multiselect(
                    'What columns fill "0" ?',
                    df_columns)
                df_columns = df_columns.difference(fill_zero_options, sort=False)


                # st.divider()
                st.subheader("Fill 1 into Null")
                fill_one_options = st.multiselect(
                    'What columns fill "1" ?',
                    df_columns)
                df_columns = df_columns.difference(fill_one_options, sort=False)
                # st.divider()

                for column in drop_null_options:
                    df_new = df_new.dropna(axis=0, subset=[column])
                    df_new.reset_index(drop=True, inplace=True)

                for column in fill_mean_options:
                    df_new[column] = df_new[column].fillna(df_new[column].mean())

                for column in fill_median_options:
                    df_new[column] = df_new[column].fillna(df_new[column].median())

                for column in fill_zero_options:
                    df_new[column] = df_new[column].fillna(0)

                for column in fill_one_options:
                    df_new[column] = df_new[column].fillna(1)

        # st.write("")
        # st.divider()
        # st.write("")

    #Category Column 생성기 만들기 <분류_로지스틱회귀/KNN/나이브베이즈_과제(타이타닉) 참고하기>


        st.header('One Hot Encoding')
        one_hot_encoding_columns = st.multiselect(
            'What columns One_Hot_Encoding?',
            df_columns)

        df_new = pd.get_dummies(df_new, columns=one_hot_encoding_columns)
        df_new_columns = df_new.columns


        # st.divider()

        st.header('Columns Selection')
        new_columns_options = st.multiselect(
            'What columns you will use?',
            df_new_columns,
            list(df_new_columns))
        df_new = df_new[new_columns_options]

    st.header('After : Detail of the Data')
    tab2_1, tab2_2, tab2_3 = st.tabs(["Information", "Description", "Correlation"])

    with tab2_1:
        st.header("Information")
        buffer2 = io.StringIO()
        df_new.info(buf=buffer2)
        df_info2 = buffer2.getvalue()
        st.text(df_info2)

    with tab2_2:
        st.header("Description")
        description_options2 = st.selectbox(
        'Select Data Type',
        ['Numeric', 'Category', 'All'])
        if description_options2 == 'Numeric':
            try:
                df_new_describe = df_new.describe(exclude='object')
            except:
                st.write("No Numerical Data")
        elif description_options2 == 'Category':
            try:
                df_new_describe = df_new.describe(include='object')
            except:
                st.write("No Categorical Data")
        else:
            df_new_describe = df_new.describe(include='all')
        try:
            st.write(df_new_describe)
        except:
            st.write('')


    with tab2_3:
        st.header("Correlation")
        fig, ax = plt.subplots()
        sns.heatmap(df_new.corr(numeric_only=True), ax=ax, cmap='Reds', annot=True, annot_kws={"size": 7})
        st.write(fig)

    # st.divider()

    st.header('Supervised Learning Setting')
    target_variable = st.selectbox(
    'Which column is Target Variable?',
    df_new.columns)
    X = df_new[df_new.columns.difference([target_variable])]
    y = df_new[target_variable]

    st.write(y.value_counts())

    sampling = st.selectbox(
        'Do You Need Sampling Operation?',
        ('No Need Sampling', 'OverSampling', 'UnderSampling'))

    ros = RandomOverSampler()
    rus = RandomUnderSampler()

    if sampling == 'OverSampling':
        X_ros, y_ros = ros.fit_resample(X, y)
        X = X_ros
        y = y_ros
        st.write('After OverSampling')
        st.write(y.value_counts())

    elif sampling == 'UnderSampling':
        X_rus, y_rus = rus.fit_resample(X, y)
        X = X_rus
        y = y_rus
        st.write('After UnderSampling')
        st.write(y.value_counts())


    with st.expander("Operate Machine Learning"):
        st.title('Machine Learning Operator')

        # st.divider()

        # st.header('Supervised Learning Setting')
        # target_variable = st.selectbox(
        # 'Which column is Target Variable?',
        # df_new.columns)
        # X = df_new[df_new.columns.difference([target_variable])]
        # y = df_new[target_variable]

        # st.divider()

        st.subheader('Test Size Setting')
        test_set_size = st.slider('Choose % of Test Set Size?', 0, 100, 20, 5)
        st.write("Your Train Set Size is", 100 - test_set_size, '%')
        st.write("Your Test Set Size is", test_set_size, '%')

        test_size = test_set_size / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        # st.divider()

        st.subheader('Train & Test Data Sizes')
        st.write('Train Data Set Size is ', X_train.shape, y_train.shape)
        st.write('Test Data Set Size is ', X_test.shape, y_test.shape)

        # st.divider()

        st.header('Task Selection')
        task_option = st.selectbox(
            'Choose Task',
            ('None', 'Regression', 'Classification'))

        # st.divider()

        if task_option == 'Regression':

            st.header('Model Selection')
            model_selection = st.selectbox(
                'Choose Machine Learning Model?',
                ['None', 'Linear Regression', 'Polinomial Regression'])

            if model_selection == 'Linear Regression':
                model = LinearRegression()
                X_train_model = X_train
                X_test_model = X_test

            elif model_selection == 'Polinomial Regression':
                poly_degree_size = st.slider('Choose Degree for Polinomial?', 0, 5, 3, 1)
                poly = PolynomialFeatures(degree=poly_degree_size)
                X_train_model = poly.fit_transform(X_train)
                X_test_model = poly.fit_transform(X_test)
                model = LinearRegression()

            # st.divider()

            try:
                st.header('Model Training & Parameter Check')
                model.fit(X=X_train_model, y=y_train)
                st.write("Model's intercept is", model.intercept_)
                st.write("Model's coefficient is/are", list(model.coef_))

                # st.divider()

                st.header('Model Test')
                y_pred = model.predict(X_test_model)
                st.write("Mean Absolute Error is", mean_absolute_error(y_test, y_pred))
                st.write("R^2 score is", r2_score(y_test, y_pred))
            except:
                st.write('Choose Model')

            # st.divider()

            if len(X.columns) == 1:
                st.header('Model Plot')

                fig = plt.figure(figsize=(12,4))
                plt.plot(X_test, y_test, 'bo')
                plt.plot(X_test, y_pred, 'r+')
                plt.xlabel(X.columns[0])
                plt.ylabel(target_variable)
                st.pyplot(fig)

        elif task_option == 'Classification':

            st.header('Model Selection')
            model_selection = st.selectbox(
                'Choose Machine Learning Model?',
                ['None', 'Logistic Regression', 'KNN(K Nearest Neighbor)', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Ada Boost', 'Gradient Boosting'])

            if model_selection == 'Logistic Regression':
                model = LogisticRegression()
                X_train_model = X_train
                X_test_model = X_test

            elif model_selection == 'KNN(K Nearest Neighbor)':
                n_neighbors = st.slider('Choose Number of Neighbor?', 1, 9, 3, 2)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
                X_train_model = X_train
                X_test_model = X_test

            elif model_selection == 'Naive Bayes':
                model = CategoricalNB(alpha=0)
                X_train_model = X_train
                X_test_model = X_test

            elif model_selection == 'Decision Tree':
                criterion = st.selectbox(
                    'Choose Decision Tree Criterion?',
                    ['entropy', 'gini', 'log_loss'])
                max_depth = st.slider('Choose Max Depth for DT?', 1, 9, 7, 1)
                min_samples_leaf = st.slider('Choose Min_Samples_Leaf for DT?', 1, 20, 12, 1)
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                X_train_model = X_train
                X_test_model = X_test

            elif model_selection == 'Random Forest':
                criterion = st.selectbox(
                    'Choose Random Forest Criterion?',
                    ['entropy', 'gini', 'log_loss'])
                max_depth = st.slider('Choose Max Depth for RF?', 1, 9, 7, 1)
                min_samples_leaf = st.slider('Choose Min_Samples_Leaf for DT?', 1, 20, 12, 1)
                max_samples = st.slider('Choose Max Sample Rate for RF?', 0.1, 1.0, 0.5, 0.1)
                model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_samples=max_samples)
                X_train_model = X_train
                X_test_model = X_test

            elif model_selection == 'Ada Boost':
                model = AdaBoostClassifier()
                X_train_model = X_train
                X_test_model = X_test

            elif model_selection == 'Gradient Boosting':
                model = GradientBoostingClassifier()
                X_train_model = X_train
                X_test_model = X_test

            # st.divider()

            try:
                st.header('Model Training & Parameters Check')
                model.fit(X=X_train_model, y=y_train)
                if (model_selection != 'KNN(K Nearest Neighbor)') & (model_selection != 'Naive Bayes') & (model_selection != 'Decision Tree') & (model_selection != 'Random Forest') & (model_selection != 'Ada Boost') & (model_selection != 'Gradient Boosting'):
                    st.write("Model's intercept is", list(model.intercept_))
                    st.write("Model's coefficient is/are", list(model.coef_))
                else:
                    st.write("This model does not have intercept & coefficient")

                if (model_selection == 'Decision Tree'):
                    max_depth_plot = st.slider('Choose Max Depth for DT?', 1, max_depth, 2, 1)
                    fig2 = plt.figure(figsize=(15,8))
                    plot_tree(model, feature_names=X_train.columns, max_depth=max_depth_plot)
                    st.pyplot(fig2)

                # st.divider()

                st.header('Model Test')
                y_pred = model.predict(X_test_model)
                st.write("Confusion Matrix is", confusion_matrix(y_test, y_pred))
                st.write("Classification_Report is")
                code = classification_report(y_test, y_pred)
                st.text('>' + code)

                # st.divider()

                st.header('Model Test Plot')

                col1_1, col1_2 = st.columns(2)

                with col1_1:
                    st.subheader("Precision-Recall Curve")
                    precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
                    fig3 = plt.figure(figsize=(5,3))
                    plt.plot(precision, recall, label=f"{model_selection}'s Precision-Recall Curve")
                    plt.legend()
                    plt.xlabel('precision')
                    plt.ylabel('recall')
                    st.pyplot(fig3)


                with col1_2:
                    st.subheader("ROC Curve")
                    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
                    fig4 = plt.figure(figsize=(5,3))
                    plt.plot(fpr, tpr, color='orange', label=f"{model_selection}'s ROC Curve")
                    plt.legend()
                    plt.xlabel('FPR')
                    plt.ylabel('TPR')
                    st.pyplot(fig4)
            except:
                st.write('Choose Model')

            # st.divider()


# 군집화 , PCA, 자연어, 비지도학습, 추천시스템




    with st.expander("Operate Deep Learning"):

        st.title('Deep Learning Operator')

        # st.divider()

        st.header('Train, Validation, Test Size Setting')
        # test_set_size = st.slider('Choose % of Test Set Size for DL?', 0, 100, 20, 5)
        # st.write("Your Train Set Size is", 100 - test_set_size, '%')
        # st.write("Your Test Set Size is", test_set_size, '%')

        val_set_size, test_set_size = st.select_slider(
            'Choose % of Train, Validation, Test Set Size for DL?',
            options=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            # value=('Validation Set Start Point', 'Test Set Start Point'))
            value=(80, 90))
        st.write('Your Train Set Size is', val_set_size, '%')
        st.write('Your Validation Set Size is', test_set_size - val_set_size, '%')
        st.write('Your Test Set Size is', 100 - test_set_size, '%')

        val_size = val_set_size / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=val_size, random_state=0)

        validation_and_test = 100 - val_set_size
        validation_to_test = test_set_size - val_set_size
        validation_rate = validation_to_test / validation_and_test
        X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, train_size=validation_rate, random_state=0)

        # st.divider()

        st.subheader('Train & Validation & Test Data Sizes')
        st.write('Train Data Set Size is ', X_train.shape, y_train.shape)
        st.write('Validation Data Set Size is ', X_validation.shape, y_validation.shape)
        st.write('Test Data Set Size is ', X_test.shape, y_test.shape)

        # st.divider()
        st.header('Task Selection')
        task_option = st.selectbox(
            'Choose Task for DL',
            ('None', 'Regression', 'Classification'))

        # st.divider()

        if task_option == 'Classification':
            st.subheader('Building Model')
            input_features = X_train.shape[1]
            # st.write(input_features)

            first_value = 1
            recommended_value = 0

            while recommended_value < input_features:
                recommended_value = 2 ** first_value
                first_value += 1

            # st.write(recommended_value)
            dl_hidden_size = st.slider("Choose First Hidden Layer's Number of Nodes?", 0, 512, recommended_value, 2)
            first_hidden_activation = st.selectbox(
            'Select Activation for First Hidden Layer',
            ['sigmoid', 'relu', 'softmax', 'tanh'])
            dl_output_size = df_new[target_variable].nunique()
            # st.write(dl_output_size)
            if dl_output_size >= 3:
                output_activation = 'softmax'
                compile_loss = 'categorical_crossentropy'

            elif dl_output_size <= 2:
                dl_output_size = 1
                output_activation = 'sigmoid'
                compile_loss = 'binary_crossentropy'

            # tensorflow model
            x = Input(shape=input_features)
            h1 = Dense(dl_hidden_size, activation=first_hidden_activation)(x)
            output = Dense(dl_output_size, activation=output_activation)(h1)
            mlp_model = tf.keras.Model(x, output)

            mlp_model.summary(print_fn=lambda x: st.text(x))

            st.subheader('Model Compile Section')
            optimizer = st.selectbox(
            'Select Optimizer for Model',
            ['adam', 'rmsprop', 'adagrad', 'momentum', 'sgd'])

            if optimizer == 'momentum':
                momentum = st.slider('Choose momentum rate?', 0.1, 1.0, 0.9, 0.1)
                optimizer = tf.keras.optimizers.SGD(momentum=momentum)

            mlp_model.compile(optimizer=optimizer, loss=compile_loss, metrics=['accuracy'])

            st.subheader('Batch_Size & Epoch Setting')

            batch_size = st.select_slider(
                'Choose Batch Size for DL?',
                options=[32, 64, 128, 256, 512],
                value=(32))

            epochs = st.slider('Choose Epochs Size?', 10, 500, 10, 10)

            history = mlp_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validation, y_validation), verbose=0)

            st.header('Train vs Validation Graph')

            col2_1, col2_2 = st.columns(2)

            with col2_1:
                fig5 = plt.figure(figsize=(5,4))
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.legend(['accuracy', 'val_accuracy'])
                plt.title('Accuracy Graph')
                plt.xlabel('Epoch')
                # plt.ylim([0, 1.5])
                plt.ylabel('Accuracy')
                st.pyplot(fig5)

            with col2_2:
                fig6 = plt.figure(figsize=(5,4))
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.legend(['loss', 'val_loss'])
                plt.title('Loss Graph')
                plt.xlabel('Epoch')
                # plt.ylim([0, 1.5])
                plt.ylabel('Loss')
                st.pyplot(fig6)


            st.header('Model Test')
            y_pred = (mlp_model.predict(X_test) > 0.5).astype(int)
            st.write("Confusion Matrix is", confusion_matrix(y_test, y_pred))
            st.write("Classification_Report is")
            code = classification_report(y_test, y_pred)
            st.text('>' + code)
