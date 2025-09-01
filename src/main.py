import os
# import json # Removed as JSON file output is no longer needed
from windbg_interface import WindbgInterface, load_config
from chunking import chunk_text

# --- Configuration Flags ---
ENABLE_CHUNKING = True  # Set to False to disable chunking functionality
# ---------------------------

def main():
    print("🚀 Windbg Analysis Tool - Multi-Command Output Capture & Chunking 🚀")
    cdb_path, minidump_path, chonkie_api_key = load_config()
    if not cdb_path or not minidump_path:
        print("❌ Error: CDB_PATH or MINIDUMP_PATH not set in .env file. ❌")
        return
    
    if ENABLE_CHUNKING and not chonkie_api_key:
        print("❌ Error: CHONKIE_API_KEY not set in .env file, but chunking is enabled. ❌")
        return

    debugger = WindbgInterface(cdb_path, minidump_path)

    minidump_filename = os.path.basename(minidump_path)
    minidump_date = minidump_filename.split('-')[0] if '-' in minidump_filename else "unknown_date"

    base_output_dir = os.path.join("command_outputs", minidump_date)
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"✅ Created output directory: {base_output_dir}")
    
    commands_to_run = [
        "kv",
        "!thread",
        "!devstack"
    ]

    print("\n--- ⏳ Starting Multi-Command Execution ⏳ ---")
    for command in commands_to_run:
        print(f"\n▶️ Executing command: '{command}'")
        sanitized_command_name = ''.join(c if c.isalnum() else '_' for c in command)
        
        # --- Save Raw Parsed Output ---
        output_file_name = f"{sanitized_command_name}.txt"
        output_file_path = os.path.join(base_output_dir, output_file_name)

        parsed_output = debugger.execute_command_and_parse_output(command, timeout=90)

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(parsed_output)
            
        # print(f"💾 Parsed output for '{command}' written to {output_file_path}")
        # --------------------------------

        # --- Chunking Functionality ---
        if ENABLE_CHUNKING:
            print(f"✨ Chunking output for '{command}' using Chonkie Cloud...")
            try:
                chunks_data = chunk_text(parsed_output, chonkie_api_key)
                
                #print amount of chunks for the command ->
                print(f"📦 Chunks for '{command}': {len(chunks_data)}")
                #then print chunks by text value -> not  the embeddings
                for chunk in chunks_data:
                    print(f"📦 Chunk for '{command}': {chunk.text}")
                
            except Exception as e:
                print(f"❌ Error chunking output for '{command}': {e}")
        # --------------------------------
    
    print("\n--- 🎉 All Commands Executed and Outputs Saved 🎉 ---")

if __name__ == "__main__":
    main()
