import os
import sys
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
from pathlib import Path

from windbg_interface import WindbgInterface, load_config
from langchain_interface import VectorStoreManager, LLMInterface
from prompt_templates import PromptTemplateManager
from app_logger import logger


class DebuggingPipeline:    
    def __init__(self, cdb_path: str, minidump_path: str):
        """Initialize the debugging pipeline."""
        self.debugger = WindbgInterface(cdb_path, minidump_path)
        self.vector_manager = VectorStoreManager()
        self.llm = LLMInterface()
        self.prompt_manager = PromptTemplateManager()

        # pipeline state
        self.analyze_v_output = ""
        self.command_outputs: List[Tuple[str, str]] = []  # (command, output)
        self.used_commands: List[str] = []
        self.citations: List[str] = []
        
        # output dir for command outputs and final analysis
        minidump_filename = os.path.basename(minidump_path)
        minidump_date = minidump_filename.split('-')[0] if '-' in minidump_filename else "unknown_date"
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("command_outputs") / f"{run_timestamp}-{minidump_date}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Logging initialized. Output directory: {self.output_dir}")
        
        self.llm_interaction_count = 0
    
    def _log_llm_interaction(self, interaction_type: str, prompt: str, response: str, 
                            metadata: Dict[str, Any] = None) -> None:
        self.llm_interaction_count += 1
        
        # Log only essential information to keep logs clean
        logger.info(f"LLM interaction #{self.llm_interaction_count} ({interaction_type})")
        if metadata:
            logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        # For final analysis, just log that it's complete - don't duplicate the full response
        if interaction_type == "final_analysis":
            logger.info("Final analysis complete - saved to JSON files")
        else:
            # Log only the response for non-final interactions
            logger.info("LLM Response:")
            logger.info("-"*40)
            logger.info(response)
    
    def run_initial_analyze(self) -> bool:

        logger.info("Starting analyze -v")
        
        # Execute !analyze -v
        self.analyze_v_output = self.debugger.execute_command_and_parse_output("!analyze -v", timeout=120)
        
        # Save output
        with open(self.output_dir / "analyze_v.txt", "w", encoding="utf-8") as f:
            f.write(self.analyze_v_output)
        
        # Check if bugcheck is authorized
        if self.prompt_manager.is_unauthorized_bugcheck(self.analyze_v_output):
            logger.error("Unauthorized bugcheck code detected")
            return False
        
        logger.info("Authorized bugcheck code detected")
        
        # Chunk and store initial analyze -v
        logger.info("Chunking analyze -v output...")
        try:
            chunks = self.vector_manager.chunk_text(self.analyze_v_output)
            documents, document_uuids = self.vector_manager.create_documents_from_command_chunks(
                chunks, "!analyze -v"
            )
            self.vector_manager.add_documents(documents)
            logger.info(f"Stored {len(chunks)} chunks from analyze -v (Total docs: {self.vector_manager.get_total_document_count()})")
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Error chunking analyze -v: {e}")
            return False
        
        return True
    
    def run_initial_prompt(self) -> Tuple[str, str]:
        """
        Step 2: Create and run initial prompt.
        Returns the command alias and argument tuple.
        """
        logger.info("Running initial prompt...")
        
        # Create initial prompt
        initial_prompt = self.prompt_manager.get_initial_prompt(self.analyze_v_output)
        
        # Get LLM response
        response = self.llm.invoke(initial_prompt)
        
        # Log the complete interaction
        metadata = {
            "bugcheck_authorized": not self.prompt_manager.is_unauthorized_bugcheck(self.analyze_v_output),
            "prompt_length": len(initial_prompt),
            "response_length": len(response)
        }
        self._log_llm_interaction("initial", initial_prompt, response, metadata)
        
        # Validate response format - no RAG query expected for initial prompt
        expected_sections = ['CITATIONS', 'COMMAND', 'CMD_ARGUMENT']
        validation_results = self.prompt_manager.validate_response_format(response, expected_sections)
        
        missing_sections = [section for section, found in validation_results.items() if not found]
        if missing_sections:
            logger.debug(f"Missing delimiter sections in initial response: {missing_sections}")
        
        # Extract individual citations, command alias, and argument
        individual_citations = self.prompt_manager.extract_individual_citations_from_response(response)
        if individual_citations:
            self.citations.extend(individual_citations)  # Add each citation separately
            logger.info(f"Collected {len(individual_citations)} citations from initial response")
        
        command_alias = self.prompt_manager.extract_command_from_response(response)
        command_argument = self.prompt_manager.extract_command_argument_from_response(response)
        
        return command_alias, command_argument
    
    def validate_and_execute_command(self, command_alias: str, argument: str) -> Tuple[str, str]:
        """
        Validate command arguments and execute if valid.
        Returns (validated_command, output) or retries with different command.
        """
        if command_alias == self.prompt_manager.stop_command:
            return command_alias, ""
        
        # Validate command arguments using new spec
        max_retries = 3
        for attempt in range(max_retries):
            if self.prompt_manager.validate_command_arguments(command_alias, argument, self.vector_manager):
                logger.info(f"Command validation passed: {command_alias} with argument: {argument}")
                break
            else:
                logger.warning(f"Command validation failed (attempt {attempt + 1}): {command_alias} with argument: {argument}")
                if attempt < max_retries - 1:
                    # Remove this command from available commands and rerun prompt
                    self.used_commands.append(command_alias)
                    # Create a new prompt without this command (no RAG context for retry)
                    new_prompt = self.prompt_manager.get_main_loop_prompt(
                        self.analyze_v_output, self.command_outputs, None, self.used_commands
                    )
                    response = self.llm.invoke(new_prompt)
                    
                    # Log the retry interaction
                    retry_metadata = {
                        "retry_attempt": attempt + 1,
                        "failed_command": command_alias,
                        "failed_argument": argument,
                        "available_commands": self.prompt_manager.get_command_count() - len(self.used_commands)
                    }
                    self._log_llm_interaction(f"retry_{attempt+1}", new_prompt, response, retry_metadata)
                    
                    command_alias = self.prompt_manager.extract_command_from_response(response)
                    argument = self.prompt_manager.extract_command_argument_from_response(response)
                else:
                    logger.warning("Max retries reached, using STOP_DEBUGGING")
                    return self.prompt_manager.stop_command, ""
        
        # Reconstruct the actual command
        if command_alias == self.prompt_manager.stop_command:
            return command_alias, ""
        
        actual_command = self.prompt_manager.reconstruct_command(command_alias, argument)
        logger.info(f"Reconstructed command: {actual_command}")
        
        # Execute the reconstructed command
        output = self.debugger.execute_command_and_parse_output(actual_command, timeout=90)
        
        # Save command output
        sanitized_command = ''.join(c if c.isalnum() else '_' for c in actual_command)
        output_file = self.output_dir / f"{sanitized_command}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        
        # Chunk and store command output
        try:
            chunks = self.vector_manager.chunk_text(output)
            documents, document_uuids = self.vector_manager.create_documents_from_command_chunks(
                chunks, actual_command
            )
            self.vector_manager.add_documents(documents)
            logger.info(f"Stored {len(chunks)} chunks from command output (Total docs: {self.vector_manager.get_total_document_count()})")
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Error chunking command output: {e}")
        
        return actual_command, output
    
    def run_main_loop(self) -> None:
        """
        Step 3: Main debugging loop until STOP_DEBUGGING or no commands left.
        """
        logger.info("Starting main debugging loop...")
        
        # Start with initial command
        initial_command_alias, initial_argument = self.run_initial_prompt()
        
        current_command_alias = initial_command_alias
        current_argument = initial_argument
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        rag_query = "debugging context"  # Initialize for first iteration
        
        while (current_command_alias != self.prompt_manager.stop_command and 
               len(self.used_commands) < self.prompt_manager.get_command_count() and
               iteration < max_iterations):
            
            iteration += 1
            logger.info(f"\nMain Loop Iteration {iteration}")
            
            # Validate and execute command
            executed_command, command_output = self.validate_and_execute_command(current_command_alias, current_argument)
            
            if executed_command == self.prompt_manager.stop_command:
                logger.info("STOP_DEBUGGING command received")
                break
            
            # Store command output
            self.command_outputs.append((executed_command, command_output))
            self.used_commands.append(current_command_alias)  # Track by alias
            
            # For first iteration, we don't need RAG since we have all data already
            # For subsequent iterations, use RAG query from previous response
            if iteration == 1:
                # First iteration: no RAG needed, we have analyze -v + first command output
                main_prompt = self.prompt_manager.get_main_loop_prompt(
                    self.analyze_v_output, self.command_outputs, None, self.used_commands
                )
                rag_query = "debugging context"  # Default for logging
            else:
                # Subsequent iterations: use RAG query from previous response
                rag_context = self.vector_manager.retrieve_context(rag_query)
                main_prompt = self.prompt_manager.get_main_loop_prompt(
                    self.analyze_v_output, self.command_outputs, rag_context, self.used_commands
                )
            
            # Get LLM response
            response = self.llm.invoke(main_prompt)
            
            # Extract RAG query from the response for next iteration
            rag_query = self.prompt_manager.extract_rag_query_from_response(response)
            logger.info(f"RAG query extracted: {rag_query}")
            
            # Log the main loop interaction
            main_loop_metadata = {
                "iteration": iteration,
                "executed_command": executed_command,
                "command_alias": current_command_alias,
                "command_argument": current_argument,
                "used_commands_count": len(self.used_commands),
                "total_commands": self.prompt_manager.get_command_count(),
                "rag_query": rag_query,
                "citations_count": len(self.citations),
                "used_rag_retrieval": iteration > 1
            }
            self._log_llm_interaction(f"main_loop_iter_{iteration}", main_prompt, response, main_loop_metadata)
            
            # Validate response format for main loop
            expected_sections = ['CITATIONS', 'COMMAND', 'CMD_ARGUMENT', 'RAG_QUERY']
            validation_results = self.prompt_manager.validate_response_format(response, expected_sections)
            
            missing_sections = [section for section, found in validation_results.items() if not found]
            if missing_sections:
                logger.debug(f"Missing delimiter sections in main loop response: {missing_sections}")
            
            # Extract individual citations, command, and RAG query
            individual_citations = self.prompt_manager.extract_individual_citations_from_response(response)
            if individual_citations:
                self.citations.extend(individual_citations)  # Add each citation separately
                logger.info(f"Collected {len(individual_citations)} citations from iteration {iteration}")
            
            current_command_alias = self.prompt_manager.extract_command_from_response(response)
            current_argument = self.prompt_manager.extract_command_argument_from_response(response)
            
            # Check if no more commands available
            if len(self.used_commands) >= self.prompt_manager.get_command_count():
                logger.info("No more commands available, proceeding to final analysis")
                break
        
        logger.info("Main debugging loop completed")
    
    def run_final_analysis(self) -> Dict[str, Any]:
        """
        Step 4: Run final analysis and generate JSON output.
        """
        logger.info("Running final analysis...")
        
        # Create final prompt with all citations
        final_prompt = self.prompt_manager.get_final_prompt(self.citations, self.analyze_v_output)
        
        # Log the final prompt ONE TIME only
        logger.info("="*60)
        logger.info("FINAL PROMPT TO LLM:")
        logger.info("="*60)
        logger.info(final_prompt)
        logger.info("="*60)
        
        # Get final LLM response
        final_response = self.llm.invoke(final_prompt)
        
        # Log the final interaction (but not the duplicate final output)
        final_metadata = {
            "total_citations": len(self.citations),
            "total_commands_used": len(self.used_commands),
            "total_interactions": self.llm_interaction_count
        }
        # Note: Don't call _log_llm_interaction to avoid duplicate logging
        
        # Extract JSON citations and plain text analysis separately
        final_json = self.prompt_manager.extract_json_from_final_response(final_response)
        analysis_text = self.prompt_manager.extract_analysis_text_from_response(final_response)
        
        # Always save the raw response as a fallback for debugging/recovery
        raw_output_file = self.output_dir / "final_analysis_raw.txt"
        with open(raw_output_file, "w", encoding="utf-8") as f:
            f.write(final_response)
        
        # Save JSON citations
        output_file = self.output_dir / "final_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        
        # Save plain text analysis with proper newlines
        analysis_file = self.output_dir / "final_analysis.txt"
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        
        # Check if parsing was successful and log appropriate messages
        if final_json.get("citations"):
            logger.info(f"Final citations saved to: {output_file}")
        else:
            logger.warning(f"No citations extracted - check raw output in: {raw_output_file}")
            
        if analysis_text and analysis_text != "Unable to extract analysis text from LLM response":
            logger.info(f"Final analysis saved to: {analysis_file}")
        else:
            logger.warning(f"Analysis text extraction failed - check raw output in: {raw_output_file}")
        
        logger.info(f"Raw LLM response saved to: {raw_output_file}")
        
        # Return both for compatibility
        return {
            "citations": final_json.get("citations", []),
            "analysis_file": str(analysis_file),
            "raw_output_file": str(raw_output_file),
            "analysis_preview": analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete debugging pipeline.
        Returns the final analysis JSON.
        """
        start_time = datetime.now()
        logger.info("Starting Windows Crash Debugging Pipeline")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Step 1: Initial analyze -v
            if not self.run_initial_analyze():
                # Unauthorized bugcheck
                unauthorized_result = {
                    "citations": [],
                    "analysis": "This crash type is not compatible. Agent only handles Driver IRQL violations (0xA, 0xD1, 0xC4, 0xC9) and Page Fault crashes (0x50, 0x7E, 0x8E, 0x1E)."
                }
                output_file = self.output_dir / "final_analysis.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(unauthorized_result, f, indent=2)
                logger.warning("Unauthorized crash type detected, stopping analysis")
                return unauthorized_result
            
            # Step 2 & 3: Main debugging loop
            self.run_main_loop()
            
            # Step 4: Final analysis
            final_result = self.run_final_analysis()
            
            # Log pipeline completion summary
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Debugging pipeline completed successfully!")
            logger.info(f"Pipeline Summary:")
            logger.info(f"   - Duration: {duration}")
            logger.info(f"   - Total LLM interactions: {self.llm_interaction_count}")
            logger.info(f"   - Commands executed: {len(self.used_commands)}")
            logger.info(f"   - Citations collected: {len(self.citations)}")
            logger.info(f"   - Output files saved to: {self.output_dir}")
            
            return final_result
            
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            # Return a basic error result instead of raising
            return {
                "citations": [],
                "analysis": f"Pipeline failed with error: {str(e)}"
            }
        finally:
            # Ensure cleanup of resources
            self._cleanup_resources()
    
    def _cleanup_resources(self):
        """Clean up any resources to ensure proper program termination."""
        try:
            # Close debugger if it has any open handles
            if hasattr(self.debugger, 'cleanup'):
                self.debugger.cleanup()
            
            # Clear vector store to free memory
            if hasattr(self.vector_manager, 'vectorstore') and self.vector_manager.vectorstore:
                self.vector_manager.vectorstore = None
                
            logger.debug("Resources cleaned up successfully")
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")
            error_result = {
                "citations": [],
                "analysis": f"Pipeline error occurred: {str(e)}"
            }
            return error_result


def main():
    """Main entry point."""
    logger.info("Windows Crash Debugging Tool - New Pipeline Structure")
    
    # Load configuration
    cdb_path, minidump_path = load_config()
    
    if not cdb_path or not minidump_path:
        logger.error("Error: CDB_PATH or MINIDUMP_PATH not set in .env file.")
        return
    
    try:
        # Initialize and run pipeline
        pipeline = DebuggingPipeline(cdb_path, minidump_path)
        final_result = pipeline.run()
        
        # Simple completion message
        logger.info("Analysis pipeline completed successfully")
        logger.info(f"Results saved to: {pipeline.output_dir}")
        
        # Explicitly exit to ensure program terminates
        logger.info("Exiting program...")
        sys.exit(0)  # Force clean exit
        
    except (ValueError, RuntimeError, AttributeError) as e:
        logger.error(f"Fatal error: {e}")
        logger.info("Exiting due to error...")
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()
