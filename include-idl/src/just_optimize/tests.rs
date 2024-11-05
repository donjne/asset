#[cfg(all(test, feature = "parse", not(feature = "rayon-optimize")))]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::just_optimize::*;

    #[test]
    fn test_documentation_removal() {
        let test_idl = json!({
            "name": "test_program",
            "docs": ["This should be removed"],
            "description": "This should also be removed",
            "instructions": [
                {
                    "name": "initialize",
                    "docs": ["This instruction doc should be removed"],
                    "accounts": []
                }
            ],
            "errors": [
                {
                    "code": 100,
                    "msg": "This error message should be preserved"
                }
            ]
        });

        let config = OptimizationConfig::default();
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        // Verify docs were removed
        assert!(processed.get("docs").is_none());
        assert!(processed.get("description").is_none());
        assert!(processed.get("instructions")
            .and_then(|i| i.as_array())
            .and_then(|i| i.first())
            .and_then(|i| i.get("docs"))
            .is_none());

        // Verify metrics
        assert!(metrics.original_size > metrics.final_size);
        assert!(metrics.size_reduction.contains_key(&OptimizationType::DocumentationRemoval));
        assert!(metrics.items_affected.get(&OptimizationType::DocumentationRemoval).unwrap() > &0);
    }

    #[test]
    fn test_preserved_fields() {
        let mut config = OptimizationConfig::default();
        config.preserve_field("special_docs");

        let test_idl = json!({
            "docs": ["Remove this"],
            "special_docs": ["Keep this"],
        });

        let mut optimizer = IdlOptimizer::new(config);
        let (processed, _) = optimizer.optimize(test_idl).unwrap();

        assert!(processed.get("docs").is_none());
        assert!(processed.get("special_docs").is_some());
    }

    #[test]
    fn test_preserved_context() {
        let mut config = OptimizationConfig::default();
        config.preserve_context("special_section");

        let test_idl = json!({
            "regular_section": {
                "docs": ["Remove this"]
            },
            "special_section": {
                "docs": ["Keep this"]
            }
        });

        let mut optimizer = IdlOptimizer::new(config);
        let (processed, _) = optimizer.optimize(test_idl).unwrap();

        assert!(processed
            .get("regular_section")
            .unwrap()
            .get("docs")
            .is_none());
        assert!(processed
            .get("special_section")
            .unwrap()
            .get("docs")
            .is_some());
    }

    #[test]
    fn test_type_deduplication() {
        let test_idl = json!({
            "name": "test_program",
            "types": [
                {
                    "name": "UserAccount",  // Different name, same structure
                    "type": "struct",
                    "fields": [
                        {"name": "balance", "type": "u64"},
                        {"name": "owner", "type": "pubkey"}
                    ]
                },
                {
                    "name": "VaultAccount", // Different name, same structure
                    "type": "struct",
                    "fields": [
                        {"name": "balance", "type": "u64"},
                        {"name": "owner", "type": "pubkey"}
                    ]
                }
            ]
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::TypeDeduplication);
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        // Get the types array
        let types = processed.get("types")
            .expect("types array should exist")
            .as_array()
            .expect("types should be an array");

        // Still have two entries
        assert_eq!(types.len(), 2, "Should preserve array length");

        // Verify one is original and one is reference
        let (originals, references): (Vec<_>, Vec<_>) = types.iter()
            .partition(|t| t.get("type_ref").is_none());

        assert_eq!(originals.len(), 1, "Should have one original type");
        assert_eq!(references.len(), 1, "Should have one reference");

        // Verify the original type structure
        let original = originals[0];
        assert_eq!(original.get("type").and_then(|t| t.as_str()), 
            Some("struct"), "Original should be a struct");
        assert!(original.get("fields").is_some(), 
            "Original should have fields");

        // Verify the reference
        let reference = references[0];
        assert!(reference.get("type_ref").is_some(), 
            "Reference should have type_ref field");

        // Verify metrics
        assert!(metrics.size_reduction
            .get(&OptimizationType::TypeDeduplication)
            .map_or(false, |&reduction| reduction > 0),
            "Should have positive size reduction");
        
        assert_eq!(
            metrics.items_affected
                .get(&OptimizationType::TypeDeduplication)
                .copied()
                .unwrap_or(0),
            1,
            "Should have deduped one type"
        );
    }

    #[test]
    fn test_nested_type_deduplication() {
        let test_idl = json!({
            "name": "test_program",
            "instructions": [
                {
                    "name": "initialize",
                    "accounts": [
                        {
                            "name": "user",
                            "type": "struct",
                            "fields": [
                                {"name": "owner", "type": "pubkey"}
                            ]
                        }
                    ]
                },
                {
                    "name": "process",
                    "accounts": [
                        {
                            "name": "admin",
                            "type": "struct",
                            "fields": [
                                {"name": "owner", "type": "pubkey"}
                            ]
                        }
                    ]
                }
            ]
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::TypeDeduplication);
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        // Verify nested types were deduplicated
        let instructions = processed.get("instructions").unwrap().as_array().unwrap();
        let second_instruction = &instructions[1];
        
        assert!(
            second_instruction
                .get("accounts")
                .unwrap()
                .as_array()
                .unwrap()[0]
                .get("type_ref")
                .is_some(),
            "Duplicate nested type should be replaced with a reference"
        );
    }

    #[test]
    fn test_unique_types_preserved() {
        let test_idl = json!({
            "name": "test_program",
            "types": [
                {
                    "name": "UserAccount",
                    "type": "struct",
                    "fields": [
                        {"name": "owner", "type": "pubkey"}
                    ]
                },
                {
                    "name": "AdminAccount",
                    "type": "struct",
                    "fields": [
                        {"name": "owner", "type": "pubkey"},
                        {"name": "permissions", "type": "bytes"}  // Different fields from UserAccount
                    ]
                }
            ]
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::TypeDeduplication);
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        // Get the types array
        let types = processed.get("types")
            .expect("types array should exist")
            .as_array()
            .expect("types should be an array");

        // Verify both original types are preserved
        assert_eq!(types.len(), 2, "Should have two distinct types");
        
        // Verify both are original types (not references)
        for (idx, t) in types.iter().enumerate() {
            assert!(t.get("type_ref").is_none(), 
                "Type at index {} should not be a reference", idx);
            assert!(t.get("type").is_some(), 
                "Type at index {} should have 'type' field", idx);
            assert!(t.get("fields").is_some(), 
                "Type at index {} should have 'fields'", idx);
        }

        // Verify no size reduction for unique types
        assert!(metrics.size_reduction
            .get(&OptimizationType::TypeDeduplication)
            .map_or(true, |&reduction| reduction == 0),
            "No size reduction expected for unique types");
    }

    #[test]
    fn test_string_minification_basic() {
        let test_idl = json!({
            "name": "test_program",
            "string_field": "This    is  a    test    description    with    extra    spaces",
            "multiline": "Line 1\n   Line 2\n     Line 3",
            "errors": [
                {
                    "code": 100,
                    "msg": "Error    message    with    spaces"  // Should not be minified
                }
            ]
        });
    
        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::StringMinification);
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();
    
        // Check that specific field was minified
        assert_eq!(
            processed.get("string_field").unwrap().as_str().unwrap(),
            "This is a test description with extra spaces"
        );
        
        // Error message should be preserved
        assert_eq!(
            processed.get("errors").unwrap().as_array().unwrap()[0]
                .get("msg").unwrap().as_str().unwrap(),
            "Error    message    with    spaces"
        );
    
        assert!(metrics.size_reduction
            .get(&OptimizationType::StringMinification)
            .unwrap() > &0);
    }

    #[test]
    fn test_string_minification_with_max_length() {
        let test_idl = json!({
            "name": "test_program",
            "custom_field": "This is a very long description that should be truncated at a specific point",
            "other_field": "This    has    multiple    spaces",
            "msg": "This is a long error message that should not be truncated because it's in skip_minification_fields"
        });
    
        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::StringMinification)
            .set_max_string_length(20)
            .skip_minification("msg");
        
        let mut optimizer = IdlOptimizer::new(config);
        
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();
    
        // Check if strings were actually modified
        let custom_field = processed["custom_field"].as_str().unwrap();
        let other_field = processed["other_field"].as_str().unwrap();
    
        // Verify the actual changes
        assert_eq!(
            custom_field,
            "This is a very lo...",
            "Long string should be truncated correctly"
        );
    
        assert_eq!(
            other_field,
            "This has multiple...",
            "String should be both minimized and truncated"
        );
    
        assert_eq!(
            processed["msg"].as_str().unwrap(),
            "This is a long error message that should not be truncated because it's in skip_minification_fields",
            "Skip field should remain unchanged"
        );
    
        // Verify final count
        let affected_count = metrics.items_affected
            .get(&OptimizationType::StringMinification)
            .copied()
            .unwrap_or(0);
        
        println!("Final affected count: {}", affected_count);
        
        assert_eq!(
            affected_count,
            2,
            "Should have affected exactly two strings: custom_field and other_field"
        );
    }
    
    #[test]
    fn test_string_minification_utf8() {
        let test_idl = json!({
            "name": "test_program",
            "utf8_field": "Hello 👋 World! This is a long string with emoji"
        });
    
        let mut config = OptimizationConfig::default();
        config.disable(OptimizationType::DocumentationRemoval)
            .enable(OptimizationType::StringMinification)
            .set_max_string_length(20);
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, _) = optimizer.optimize(test_idl).unwrap();
    
        let result = processed.get("utf8_field").unwrap().as_str().unwrap();
        
        // Count characters, not bytes
        assert_eq!(
            result.chars().count(),
            20,
            "Result should be exactly 20 characters (including ellipsis)"
        );
        
        // Verify it ends with ellipsis
        assert!(result.ends_with("..."), "Should end with ellipsis");
        
        // Verify it's valid UTF-8 and preserves emoji
        assert!(result.contains("👋"), "Should preserve emoji");
        assert!(String::from_utf8(result.as_bytes().to_vec()).is_ok(), "Should be valid UTF-8");
    }

    #[test]
    fn test_string_minification_preserve_indentation() {
        let test_idl = json!({
            "name": "test_program",
            "code": "fn main() {   \n    let x = 1;    \n    println!(\"x: {}\", x);    \n}"
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::StringMinification)
            .preserve_indentation(true);
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, _) = optimizer.optimize(test_idl).unwrap();

        assert_eq!(
            processed.get("code").unwrap().as_str().unwrap(),
            "fn main() {\n    let x = 1;\n    println!(\"x: {}\", x);\n}"
        );
    }

    #[test]
    fn test_string_minification_skip_fields() {
        let test_idl = json!({
            "name": "test_program",
            "normal_field": "This    has    spaces",
            "preserved_field": "This    also    has    spaces"
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::StringMinification)
            .skip_minification("preserved_field");
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, _) = optimizer.optimize(test_idl).unwrap();

        assert_eq!(
            processed.get("normal_field").unwrap().as_str().unwrap(),
            "This has spaces"
        );
        assert_eq!(
            processed.get("preserved_field").unwrap().as_str().unwrap(),
            "This    also    has    spaces"
        );
    }

    #[test]
    fn test_custom_field_removal_counting() {
        let test_idl = json!({
            "name": "test_program",
            "internal_id": "remove1",           // Should be removed
            "debug_info": "remove2",            // Should be removed
            "metadata": {
                "internal_version": "remove3",  // Should be removed
                "debug_flag": "remove4",        // Should be removed
                "keep_this": "stays"
            },
            "array_field": [
                {
                    "internal_data": "remove5", // Should be removed
                    "normal_data": "stays"
                },
                {
                    "debug_mode": "remove6",    // Should be removed
                    "other_data": "stays"
                }
            ],
            "protected": {
                "internal_secret": "stays",     // Protected by context
                "debug_flag": "stays"          // Protected by context
            }
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::CustomFieldRemoval)
            .remove_field_pattern("internal_*")
            .remove_field_pattern("debug_*")
            .exempt_context("protected");
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        // Verify removals
        assert!(processed.get("internal_id").is_none());
        assert!(processed.get("debug_info").is_none());
        
        let metadata = processed.get("metadata")
            .and_then(|m| m.as_object())
            .expect("metadata should exist");
        assert!(metadata.get("internal_version").is_none());
        assert!(metadata.get("debug_flag").is_none());
        assert!(metadata.get("keep_this").is_some());

        let array_field = processed.get("array_field")
            .and_then(|a| a.as_array())
            .expect("array_field should exist");
        assert!(array_field[0].get("internal_data").is_none());
        assert!(array_field[0].get("normal_data").is_some());
        assert!(array_field[1].get("debug_mode").is_none());
        assert!(array_field[1].get("other_data").is_some());

        // Verify protected context
        let protected = processed.get("protected")
            .and_then(|p| p.as_object())
            .expect("protected section should exist");
        assert!(protected.get("internal_secret").is_some());
        assert!(protected.get("debug_flag").is_some());

        // Verify count matches exactly what was removed
        assert_eq!(
            metrics.items_affected
                .get(&OptimizationType::CustomFieldRemoval)
                .copied()
                .unwrap_or(0),
            6,  // Should have removed exactly 6 fields
            "Should count all removed fields except those in protected context"
        );

        // Verify size reduction
        assert!(
            metrics.size_reduction
                .get(&OptimizationType::CustomFieldRemoval)
                .copied()
                .unwrap_or(0) > 0,
            "Should have reduced total size"
        );
    }

    #[test]
    fn test_custom_field_removal_basic() {
        let test_idl = json!({
            "name": "test_program",
            "internal_id": "should_be_removed",
            "metadata": {
                "internal_version": "1.0",
                "public_version": "2.0"
            }
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::CustomFieldRemoval)
            .remove_field_pattern("internal_*");
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        // Verify internal fields were removed
        assert!(processed.get("internal_id").is_none());
        assert!(processed.get("metadata")
            .and_then(|m| m.as_object())
            .and_then(|m| m.get("internal_version"))
            .is_none());

        // Public fields should remain
        assert_eq!(
            processed.get("metadata")
                .and_then(|m| m.as_object())
                .and_then(|m| m.get("public_version"))
                .and_then(|v| v.as_str()),
            Some("2.0")
        );

        assert!(metrics.items_affected
            .get(&OptimizationType::CustomFieldRemoval)
            .copied()
            .unwrap_or(0) > 0);
    }

    #[test]
    fn test_custom_field_exempt_context() {
        let test_idl = json!({
            "name": "test_program",
            "internal_id": "should_be_removed",
            "protected": {
                "internal_secret": "should_be_preserved",
                "internal_key": "also_preserved"
            }
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::CustomFieldRemoval)
            .remove_field_pattern("internal_*")
            .exempt_context("protected");
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, _) = optimizer.optimize(test_idl).unwrap();

        // Verify top-level internal field was removed
        assert!(processed.get("internal_id").is_none());

        // Protected context should preserve internal fields
        let protected = processed.get("protected")
            .and_then(|p| p.as_object())
            .expect("protected section should exist");
        
        assert_eq!(
            protected.get("internal_secret").and_then(|v| v.as_str()),
            Some("should_be_preserved")
        );
        assert_eq!(
            protected.get("internal_key").and_then(|v| v.as_str()),
            Some("also_preserved")
        );
    }

    #[test]
    fn test_custom_field_patterns() {
        let test_idl = json!({
            "name": "test_program",
            "temp_var": "remove_me",
            "debug_info": "remove_me_too",
            "keepme": "should_stay",
            "_private": "remove_me_three"
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::CustomFieldRemoval)
            .remove_field_pattern("temp_*")
            .remove_field_pattern("debug_*")
            .remove_field("_private");
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        assert!(processed.get("temp_var").is_none());
        assert!(processed.get("debug_info").is_none());
        assert!(processed.get("_private").is_none());
        assert_eq!(
            processed.get("keepme").and_then(|v| v.as_str()),
            Some("should_stay")
        );

        assert_eq!(
            metrics.items_affected
                .get(&OptimizationType::CustomFieldRemoval)
                .copied()
                .unwrap_or(0),
            3
        );
    }

    #[test]
    fn test_compression_basic() {
        let test_idl = json!({
            "name": "test_program",
            "version": "1.0.0",
            "instructions": [
                {
                    "name": "initialize",
                    "accounts": [
                        {"name": "user", "type": "signer"},
                        {"name": "system", "type": "program"}
                    ],
                    "args": [
                        {"name": "data", "type": "bytes"}
                    ]
                }
            ]
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::Compression)
            .set_compression_threshold(50);  // Small threshold for testing
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        // Verify compression
        assert!(processed.get("compressed_data").is_some());
        assert!(processed.get("original_size").is_some());

        // Verify metrics
        let size_reduction = metrics.size_reduction
            .get(&OptimizationType::Compression)
            .copied()
            .unwrap_or(0);
        assert!(size_reduction > 0, "Should have reduced size through compression");
        
        assert_eq!(
            metrics.items_affected
                .get(&OptimizationType::Compression)
                .copied()
                .unwrap_or(0),
            1,
            "Should have compressed one item"
        );
    }

    #[test]
    fn test_compression_threshold() {
        let small_idl = json!({
            "name": "small",
            "version": "1.0"
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::Compression)
            .set_compression_threshold(1000);  // Larger than the test IDL
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(small_idl).unwrap();

        // Verify no compression for small data
        assert!(processed.get("compressed_data").is_none());
        
        assert_eq!(
            metrics.items_affected
                .get(&OptimizationType::Compression)
                .copied()
                .unwrap_or(0),
            0,
            "Should not compress small items"
        );
    }

    #[test]
    fn test_nested_compression() {
        let test_idl = json!({
            "name": "test_program",
            "large_nested": {
                "data": "A".repeat(1000),  // Large nested object
                "more_data": {
                    "nested": "B".repeat(1000)  // Another large nested object
                }
            }
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::Compression)
            .set_compression_threshold(500)
            .compress_nested(true);
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        assert!(
            metrics.items_affected
                .get(&OptimizationType::Compression)
                .copied()
                .unwrap_or(0) >= 2,
            "Should compress multiple nested objects"
        );
    }

    #[test]
    fn test_compression_levels() {
        let large_idl = json!({
            "data": "X".repeat(10000)  // Large string to compress
        });

        let mut sizes = Vec::new();

        // Test different compression levels
        for level in 0..=9 {
            let mut config = OptimizationConfig::default();
            config.enable(OptimizationType::Compression)
                .set_compression_level(level)
                .set_compression_threshold(0);
            
            let mut optimizer = IdlOptimizer::new(config);
            let (processed, _) = optimizer.optimize(large_idl.clone()).unwrap();
            
            let compressed = processed.get("compressed_data")
                .and_then(|v| v.as_str())
                .unwrap();
            sizes.push(compressed.len());
        }

        // Higher compression levels should generally produce smaller output
        for i in 1..sizes.len() {
            assert!(sizes[i] <= sizes[0], 
                "Higher compression levels should not increase size");
        }
    }

    #[test]
    fn test_compression_with_other_optimizations() {
        let test_idl = json!({
            "name": "test_program",
            "docs": ["This should be removed"],
            "internal_field": "This should be removed",
            "data": "A".repeat(1000)  // Large data to compress
        });

        let mut config = OptimizationConfig::default();
        config.enable(OptimizationType::DocumentationRemoval)
            .enable(OptimizationType::CustomFieldRemoval)
            .enable(OptimizationType::Compression)
            .remove_field_pattern("internal_*")
            .set_compression_threshold(100);
        
        let mut optimizer = IdlOptimizer::new(config);
        let (processed, metrics) = optimizer.optimize(test_idl).unwrap();

        // Verify all optimizations were applied
        assert!(metrics.items_affected.contains_key(&OptimizationType::DocumentationRemoval));
        assert!(metrics.items_affected.contains_key(&OptimizationType::CustomFieldRemoval));
        assert!(metrics.items_affected.contains_key(&OptimizationType::Compression));

        // Verify final result is compressed
        assert!(processed.get("compressed_data").is_some());
    }
}