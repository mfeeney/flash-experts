"""
Script to generate MTG card names using our fine-tuned model.
"""

from generate_mtg_names import CardNameGenerator, GenerationConfig

def main():
    # Initialize the generator with our trained model
    generator = CardNameGenerator(
        model_path="./fine_tuned_model/mtg_names/stage_2/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Define generation configs for different styles
    configs = {
        "Balanced": GenerationConfig(
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.5,
            name="Balanced",
            min_length=8,
            max_length=32,
            no_repeat_ngram_size=3
        ),
        "Creative": GenerationConfig(
            temperature=0.9,
            top_p=0.92,
            repetition_penalty=1.6,
            name="Creative",
            min_length=8,
            max_length=32,
            no_repeat_ngram_size=3
        ),
        "Conservative": GenerationConfig(
            temperature=0.6,
            top_p=0.8,
            repetition_penalty=1.4,
            name="Conservative",
            min_length=8,
            max_length=32,
            no_repeat_ngram_size=3
        )
    }
    
    # Card types to generate names for
    card_types = ['artifact', 'land', 'instant', 'sorcery', 'enchantment', 'creature', 'planeswalker']
    
    print("\nGenerating MTG Card Names")
    print("=" * 50)
    
    # Generate names for each card type and config
    for card_type in card_types:
        print(f"\n{card_type.upper()} NAMES:")
        print("-" * 30)
        
        for config_name, config in configs.items():
            print(f"\nUsing {config_name} generation settings:")
            names = generator.generate_names(card_type, num_batches=2, config=config)
            
            # Print generated names
            for i, name in enumerate(names, 1):
                print(f"{i}. {name}")
            
            # Analyze the generated names
            analysis = generator.analyze_names(names)
            print("\nAnalysis:")
            print(f"Average length: {analysis['average_length']:.1f} characters")
            print(f"Average words: {analysis['average_words']:.1f}")
            print("\nPattern Statistics:")
            print(f"Proper nouns: {analysis['pattern_statistics']['proper_nouns']}")
            print(f"Single-word names: {analysis['pattern_statistics']['single_word_names']}")
            print(f"Multi-word names: {analysis['pattern_statistics']['multi_word_names']}")
            print("\nMost common words:")
            for word, count in analysis['most_common_words']:
                print(f"  {word}: {count}")
            print("\n" + "-" * 30)

if __name__ == "__main__":
    import torch
    main() 