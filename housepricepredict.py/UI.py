import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# ১. ব্যাকগ্রাউন্ডে মডেল রেডি করা (ML Pipeline)
# ==========================================
# ডেটা লোড এবং ক্লিনিং
my_file = pd.read_csv('data.csv')
my_file = my_file.dropna(axis=0)

# টার্গেট এবং ফিচার আলাদা করা
y = my_file.price
file_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built']
X = my_file[file_features]

# ডেটা ভাগ করা এবং ফাইনাল মডেল ট্রেন করা
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestRegressor(random_state=0)
my_model.fit(train_X, train_y)

# ==========================================
# ২. ইউজার ইন্টারফেস তৈরি (Streamlit UI)
# ==========================================
# ওয়েবসাইটের মূল হেডার বা শিরোনাম
st.set_page_config(page_title="রিয়েল এস্টেট প্রাইস প্রেডিক্টর", page_icon="🏠")
st.title("🏠 বাড়ির দাম অনুমান করার জাদুর ক্যালকুলেটর")
st.write("নিচের বক্সে আপনার বাড়ির বিবরণ দিন এবং মুহূর্তের মধ্যে বাজার মূল্য জেনে নিন।")

st.markdown("---")  # একটি সোজা দাগ বা ডিভাইডার

# কাস্টমারের ইনপুট নেওয়ার জন্য সুন্দর বক্স ও স্লাইডার সাজানো
col1, col2 = st.columns(2)  # স্ক্রিনটিকে দুটি কলামে ভাগ করা

with col1:
    user_bedrooms = st.number_input(
        "🛏️ বেডরুমের সংখ্যা লিখুন:", min_value=1, max_value=10, value=3)
    user_bathrooms = st.number_input(
        "🚿 বাথরুমের সংখ্যা লিখুন:", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    user_sqft_living = st.number_input(
        "📐 লিভিং এরিয়ার স্কয়ার ফিট (sqft_living):", min_value=300, max_value=10000, value=2000)
    user_sqft_lot = st.number_input(
        "🌳 মোট জায়গার স্কয়ার ফিট (sqft_lot):", min_value=500, max_value=50000, value=5000)
    user_floors = st.number_input(
        "🏢 তলার সংখ্যা (Floors):", min_value=1.0, max_value=5.0, value=1.0, step=0.5)

with col2:
    user_waterfront = st.selectbox("🌊 বাড়ির পাশে কি নদী/হ্রদ আছে?", options=[
                                   0, 1], format_func=lambda x: "হ্যাঁ" if x == 1 else "না")
    user_view = st.slider(
        "👁️ বাড়ির চারপাশের ভিউ কেমন? (০ মানে খারাপ, ৪ মানে চমৎকার):", min_value=0, max_value=4, value=0)
    user_condition = st.slider(
        "🛠️ বাড়ির কন্ডিশন কেমন? (১ মানে পুরোনো, ৫ মানে একদম নতুন):", min_value=1, max_value=5, value=3)
    user_sqft_above = st.number_input(
        "📐 মাটির ওপরের অংশের স্কয়ার ফিট (sqft_above):", min_value=300, value=1700)
    user_sqft_basement = st.number_input(
        "📐 মাটির নিচের অংশ/বেসমেন্টের স্কয়ার ফিট (sqft_basement):", min_value=0, value=300)
    user_yr_built = st.number_input(
        "📅 কত সালে তৈরি হয়েছিল? (Year Built):", min_value=1900, max_value=2026, value=2000)

st.markdown("---")

# যখন কাস্টমার এই বড় গোল্ডেন বাটনে ক্লিক করবেন
if st.button("💰 বাড়ির আনুমানিক দাম হিসাব করুন", use_container_width=True):

    # কাস্টমারের দেওয়া ১১টি তথ্য দিয়ে একটি ছোট টেবিল (DataFrame) বানানো
    custom_house = pd.DataFrame([{
        'bedrooms': user_bedrooms, 'bathrooms': user_bathrooms, 'sqft_living': user_sqft_living,
        'sqft_lot': user_sqft_lot, 'floors': user_floors, 'waterfront': user_waterfront,
        'view': user_view, 'condition': user_condition, 'sqft_above': user_sqft_above,
        'sqft_basement': user_sqft_basement, 'yr_built': user_yr_built
    }])

    # মডেলের মাধ্যমে সরাসরি আসল দাম অনুমান করা
    predicted_price = my_model.predict(custom_house)

    # ফলাফলটি খুব সুন্দর করে ওয়েবসাইটের স্ক্রিনে বড় করে দেখানো
    st.success(
        f"### 🎉 আপনার বাড়ির আনুমানিক বাজার মূল্য: **${predicted_price[0]:,.2f}**")
