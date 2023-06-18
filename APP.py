# 导入需要的库
import streamlit as st
import pandas as pd
import joblib
# 页标题，显示在浏览器标签上
st.set_page_config(page_title="帕金森病预测APP",page_icon="🐳")
st.sidebar.image('机器学习.jpeg')
st.sidebar.header('机器学习模型')
choose = st.sidebar.selectbox("",('随机森林模型','朴素贝叶斯模型',
                     '逻辑回归模型'))

if choose == '随机森林模型':
    st.header("帕金森病预测系统")
    # 显示数值输入框
    # 2列布局
    left_column, right_column = st.columns(2)
    # 添加相关信息
    with left_column:
        # 显示数值输入框
        a = st.number_input("PPE")
        # 显示数值输入框
        b = st.number_input("MDVP:Flo(Hz)")
        # 显示数值输入框
        c = st.number_input("HNR")
        # 显示数值输入框
        d = st.number_input("MDVP:Fhi(Hz)")
        # 显示数值输入框
        e = st.number_input("MDVP:Fo(Hz)")

    with right_column:
        # 显示数值输入框
        f = st.number_input("spread1")
        # 显示数值输入框
        g = st.number_input("spread2")
        # 显示数值输入框
        h = st.number_input("NHR")
        # 显示数值输入框
        i = st.number_input("MDVP:Shimmer")
        # 显示数值输入框
        j = st.number_input("MDVP:RAP")
    # 如果按下按钮
    if st.button("点击预测"):  # 显示按钮
        # 加载训练好的模型
        RF = joblib.load("RF.pkl")
        # 将输入存储DataFrame
        X = pd.DataFrame([[a,b,c,d,e,f,g,h,i,j]],
                         columns = ["PPE", "MDVP:Flo(Hz)", "HNR","MDVP:Fhi(Hz)",
                                    "MDVP:Fo(Hz)", "spread1", "spread2", "NHR",
                                    "MDVP:Shimmer", "MDVP:RAP"
                                    ])

        # 进行预测
        prediction = RF.predict(X)[0]
        # 输出预测结果
        st.success(f"随机森林模型预测结果为： {prediction}")


elif choose == '朴素贝叶斯模型':
    st.header("帕金森病预测系统")
    # 显示数值输入框
    # 2列布局
    left_column, right_column = st.columns(2)
    # 添加相关信息
    with left_column:
        # 显示数值输入框
        a = st.number_input("PPE")
        # 显示数值输入框
        b = st.number_input("MDVP:Flo(Hz)")
        # 显示数值输入框
        c = st.number_input("HNR")
        # 显示数值输入框
        d = st.number_input("MDVP:Fhi(Hz)")
        # 显示数值输入框
        e = st.number_input("MDVP:Fo(Hz)")

    with right_column:
        # 显示数值输入框
        f = st.number_input("spread1")
        # 显示数值输入框
        g = st.number_input("spread2")
        # 显示数值输入框
        h = st.number_input("NHR")
        # 显示数值输入框
        i = st.number_input("MDVP:Shimmer")
        # 显示数值输入框
        j = st.number_input("MDVP:RAP")
    # 如果按下按钮
    if st.button("点击预测"):  # 显示按钮
        # 加载训练好的模型
        NB = joblib.load("NB.pkl")
        # 将输入存储DataFrame
        X = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j]],
                         columns=["PPE", "MDVP:Flo(Hz)", "HNR", "MDVP:Fhi(Hz)",
                                  "MDVP:Fo(Hz)", "spread1", "spread2", "NHR",
                                  "MDVP:Shimmer", "MDVP:RAP"
                                  ])

        # 进行预测
        prediction = NB.predict(X)[0]
        # 输出预测结果
        st.success(f"朴素贝叶斯模型预测结果为： {prediction}")

elif choose == '逻辑回归模型':
    st.header("帕金森病预测系统")
    # 显示数值输入框
    # 2列布局
    left_column, right_column = st.columns(2)
    # 添加相关信息
    with left_column:
        # 显示数值输入框
        a = st.number_input("PPE")
        # 显示数值输入框
        b = st.number_input("MDVP:Flo(Hz)")
        # 显示数值输入框
        c = st.number_input("HNR")
        # 显示数值输入框
        d = st.number_input("MDVP:Fhi(Hz)")
        # 显示数值输入框
        e = st.number_input("MDVP:Fo(Hz)")

    with right_column:
        # 显示数值输入框
        f = st.number_input("spread1")
        # 显示数值输入框
        g = st.number_input("spread2")
        # 显示数值输入框
        h = st.number_input("NHR")
        # 显示数值输入框
        i = st.number_input("MDVP:Shimmer")
        # 显示数值输入框
        j = st.number_input("MDVP:RAP")
    # 如果按下按钮
    if st.button("点击预测"):  # 显示按钮
        # 加载训练好的模型
        LR = joblib.load("LR.pkl")
        # 将输入存储DataFrame
        X = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j]],
                         columns=["PPE", "MDVP:Flo(Hz)", "HNR", "MDVP:Fhi(Hz)",
                                  "MDVP:Fo(Hz)", "spread1", "spread2", "NHR",
                                  "MDVP:Shimmer", "MDVP:RAP"
                                  ])

        # 进行预测
        prediction = LR.predict(X)[0]
        # 输出预测结果
        st.success(f"逻辑回归模型预测结果为： {prediction}")





