import torch
import gradio as gr
from model import CharacterLevelTokenizer, PotterGPT, Config

class GradioApp():
    def __init__(self):
        # Set up configuration and data
        self.model_path = 'potterGPT/potterGPT.pth'
        with open('data/harry_potter_data', 'r', encoding='utf-8') as f:
            data = f.read()

        self.tokenizer = CharacterLevelTokenizer(data)
        self.lm = PotterGPT(Config)
        state_dict = torch.load(self.model_path, map_location='cpu')
        self.lm.load_state_dict(state_dict)

    def launch(self):
        # Define Gradio interface without a clear button
        with gr.Blocks() as demo:
            gr.Markdown("# potterGPT v0")
            gr.Markdown("Click the button to generate a text prompt using the potterGPT model.")
            
            generate_button = gr.Button("Generate")
            output_text = gr.Textbox(label="Generated Text")
            
            generate_button.click(self.generate_text, inputs=None, outputs=output_text)
        
        demo.launch()

    def generate_text(self, input=None):
        """Generate text using the trained model."""
        generated_texts = []
        for length in [1000]:
            generated = self.lm.generate(
                torch.zeros((1,1),dtype=torch.long,device='cpu') + 61, # initial context 0, 61 is \n
                total=length
            )
            generated = self.tokenizer.decode(generated[0].cpu().numpy())
            text = f'generated ({length} tokens)\n{"="*50}\n{generated}\n{"="*50}\n\n'
            generated_texts.append(text)
        return generated_texts[0]

if __name__ == '__main__':
    app = GradioApp()
    app.launch()