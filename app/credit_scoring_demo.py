import gradio as gr
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
MODEL_NAME = "credit_scoring_model"
STAGE = "Production"  # or "Staging"

# Load model from MLFlow
def load_model():
    """Load the production model from MLFlow registry"""
    try:
        model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback: load from local file if MLFlow unavailable
        return None

model = load_model()

def predict_credit_risk(
    age,
    income,
    debt_ratio,
    credit_lines,
    late_payments,
    loan_amount
):
    """
    Predict credit risk based on input features
    
    Returns:
        - Risk Score (0-100)
        - Risk Category (Low/Medium/High)
        - Recommendation
    """
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'debt_ratio': [debt_ratio],
        'credit_lines': [credit_lines],
        'late_payments': [late_payments],
        'loan_amount': [loan_amount]
    })
    
    # Make prediction
    if model is not None:
        try:
            # Get probability of default
            prediction = model.predict(input_data)
            risk_score = float(prediction[0] * 100)
        except Exception as e:
            return f"Error making prediction: {e}", "", ""
    else:
        # Demo fallback (simple heuristic)
        risk_score = (
            (debt_ratio * 30) + 
            (late_payments * 15) - 
            (income / 10000) + 
            (loan_amount / 1000)
        )
        risk_score = np.clip(risk_score, 0, 100)
    
    # Categorize risk
    if risk_score < 30:
        category = "🟢 Low Risk"
        recommendation = "✅ Approve - Low risk of default"
    elif risk_score < 60:
        category = "🟡 Medium Risk"
        recommendation = "⚠️ Review - Consider additional verification"
    else:
        category = "🔴 High Risk"
        recommendation = "❌ Decline - High risk of default"
    
    return f"{risk_score:.1f}%", category, recommendation

# Create Gradio interface
with gr.Blocks(title="Credit Scoring System", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🏦 Credit Scoring System
        ### ML-Powered Credit Risk Assessment
        
        This system uses machine learning to assess credit risk based on applicant information.
        Enter the details below to get a risk assessment.
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Applicant Information")
            
            age = gr.Slider(
                minimum=18,
                maximum=80,
                value=35,
                step=1,
                label="Age (years)"
            )
            
            income = gr.Number(
                value=50000,
                label="Annual Income ($)"
            )
            
            debt_ratio = gr.Slider(
                minimum=0,
                maximum=100,
                value=30,
                step=1,
                label="Debt-to-Income Ratio (%)"
            )
            
            credit_lines = gr.Slider(
                minimum=0,
                maximum=20,
                value=3,
                step=1,
                label="Number of Credit Lines"
            )
            
            late_payments = gr.Slider(
                minimum=0,
                maximum=10,
                value=0,
                step=1,
                label="Late Payments (Last 2 Years)"
            )
            
            loan_amount = gr.Number(
                value=25000,
                label="Requested Loan Amount ($)"
            )
            
            predict_btn = gr.Button("🔍 Assess Credit Risk", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 📊 Risk Assessment")
            
            risk_score = gr.Textbox(
                label="Risk Score",
                placeholder="Click 'Assess Credit Risk' to see score"
            )
            
            risk_category = gr.Textbox(
                label="Risk Category",
                placeholder="Risk category will appear here"
            )
            
            recommendation = gr.Textbox(
                label="Recommendation",
                placeholder="Recommendation will appear here"
            )
    
    # Examples
    gr.Markdown("### 💡 Try These Examples")
    gr.Examples(
        examples=[
            [28, 45000, 25, 2, 0, 15000],  # Low risk
            [42, 75000, 45, 5, 2, 30000],  # Medium risk
            [55, 35000, 80, 8, 5, 50000],  # High risk
        ],
        inputs=[age, income, debt_ratio, credit_lines, late_payments, loan_amount],
        outputs=[risk_score, risk_category, recommendation],
        fn=predict_credit_risk,
        label="Example Scenarios"
    )
    
    # Connect button to prediction function
    predict_btn.click(
        fn=predict_credit_risk,
        inputs=[age, income, debt_ratio, credit_lines, late_payments, loan_amount],
        outputs=[risk_score, risk_category, recommendation]
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Model Info:** Credit Scoring Model trained on historical loan data  
        **Last Updated:** Check MLFlow for model version  
        **Metrics:** View performance metrics in MLFlow UI
        """
    )

if __name__ == "__main__":
    demo.launch()