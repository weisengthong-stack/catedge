#!/usr/bin/env python3

import os
from docling.document_converter import DocumentConverter
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from llama_cloud_services import LlamaExtract

# Set API key directly (choose one method)
# Method 1: Set environment variable in code
# os.environ["LLAMA_CLOUD_API_KEY"] = "llx-umPayGgM0OvbqH6sUJj9t6mgMZLOsLCAMBpSnjUGJKB2WC8J"

# Method 2: Pass API key when initializing (if supported)
# extractor = LlamaExtract(api_key="your-api-key-here")

# Define schema for customer profile extraction
# class CustomerProfile(BaseModel):
#     entityType: str = Field(description="Type of entity, e.g., ASSOCIATION.")
#     corporateWebsite: Optional[str] = Field(description="URL of the corporate website, if available.")
#     fatfjurisdiction: Optional[str] = Field(description="FATF jurisdiction associated with the customer.")
#     industry: str = Field(description="Industry sector of the customer's business, e.g., AGRICULTURE.")
#     onBoardingMode: Optional[str] = Field(description="Mode of onboarding the customer, e.g., FACE-TO-FACE.")
#     ownershipStructureLayer: Optional[str] = Field(description="Layer of the ownership structure.")
#     paymentMode: Optional[List[str]] = Field(description="Preferred payment modes of the customer.")
#     productServiceComplexity: str = Field(description='Complexity level of the products or services offered. Filled in by either "SIMPLE" or "COMPLEX"')
#     sourceOfFunds: Optional[str] = Field(description="Source of funds for the customer's transactions.")
#     natureOfBusinessRelationship: Optional[str] = Field(description="Nature of the business relationship with the customer.")
#     bankAccount: Optional[List[str]] = Field(description="List of bank accounts associated with the customer.")
#     additionalInformation: Optional[str] = Field(description="Any additional information about the customer.")
#     incorporated: bool = Field(description="Indicates whether the customer is incorporated.")
#     name: str = Field(description="Official name of the customer.")
#     alias: Optional[str] = Field(description="List of aliases or alternative names used by the customer.")
#     formerName: Optional[str] = Field(description="List of former names of the customer.")
#     countryOfIncorporation: str = Field(description="Country where the customer was incorporated.")
#     countryOfOperation: str = Field(description="List of countries where the customer operates.")
#     address: str = Field(description='List of addresses associated with the customer. if exists, take the "Address of principal place of business"')
#     incorporateNumber: str = Field(description="Incorporation number of the customer.")
#     phone: Optional[str] = Field(description="List of phone numbers associated with the customer.")
#     email: Optional[str] = Field(description="List of email addresses associated with the customer.")
#     dateOfIncorporation: Optional[str] = Field(description="Date when the customer was incorporated (format: YYYY-MM-DD).")
#     dateOfIncorporationExpiry: Optional[str] = Field(description="Expiry date of incorporation (format: YYYY-MM-DD).")
#     imonumber: Optional[str] = Field(description="IMO number of the customer, if applicable.")
#     active: bool = Field(description="Indicates whether the customer profile is currently active.")
#     profileReferenceId: str = Field(description="Unique identifier for the customer profile.")
#     other: Optional[str] = Field(description="Contains additional information about the customer, such as entity type, industry, and payment preferences.")
#     particular: Optional[str] = Field(description="Specific details about the customer, such as incorporation status, name, and contact information.")
#     type: Optional[str] = Field(description="Type of customer, e.g., CORPORATE.")
#     domainId: Optional[List[str]] = Field(description="List of domain IDs associated with the customer.")
#     assigneeId: Optional[str] = Field(description="ID of the assignee responsible for the customer.")

def main():
    # Configure input and output
    source = "joeyyap_doc.pdf"  # document per local path or URL
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Convert document using Docling
    print("Converting document with Docling...")
    converter = DocumentConverter()
    result = converter.convert(source)
    
    # Export to Markdown
    markdown_output = result.document.export_to_markdown()
    
    # Get document name for output files
    doc_name = Path(source).stem
    markdown_path = output_dir / f"{doc_name}.md"
    
    # Save markdown file
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"Markdown saved to: {markdown_path}")
    
    # # Extract structured data using LlamaExtract
    # print("\nExtracting structured customer profile data with LlamaExtract...")
    # extractor = LlamaExtract()
    
    # # Create extraction agent
    # agent = extractor.get_agent(name="customer-profile-parser")
    # # agent = extractor.create_agent(
    # #     name="customer-profile-parser",
    # #     data_schema=CustomerProfile
    # # )
    
    # # Extract customer profile data from markdown file
    # customer_data = agent.extract(markdown_path)
    
    # # Print extracted customer profile information
    # print(customer_data.data)
    
    # # # Save extracted data as JSON
    # import json
    # extracted_json_path = output_dir / f"{doc_name}_extracted.json"
    # with open(extracted_json_path, "w", encoding="utf-8") as f:
    #     json.dump(customer_data.data, f, indent=2, ensure_ascii=False)
    # print(f"\nExtracted data saved to: {extracted_json_path}")

if __name__ == "__main__":
    main()