#[cfg(feature = "parse")]
use serde_json::{Map, Value};
use std::{collections::{HashMap, HashSet}, time::Instant};
use md5::compute;
use std::io::Write;
use base64::{Engine as _, engine::general_purpose::STANDARD};

/// Types of optimizations available
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum OptimizationType {
    DocumentationRemoval,
    TypeDeduplication,
    StringMinification,
    CustomFieldRemoval,
    Compression,
}

/// Configuration for the optimization process
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Which optimizations are enabled
    enabled_optimizations: HashSet<OptimizationType>,
    /// Fields considered as documentation
    doc_fields: HashSet<String>,
    /// Fields to preserve even if they match doc patterns
    preserved_fields: HashSet<String>,
    /// Contexts where docs should be preserved (e.g., "errors")
    preserved_contexts: HashSet<String>,
    /// Fields that identify a type (default: "name", "type", "fields")
    type_identifier_fields: HashSet<String>,
    /// Whether to deduplicate nested types
    deduplicate_nested: bool,
    /// Fields to skip during string minification
    skip_minification_fields: HashSet<String>,
    /// Whether to preserve indentation in certain strings
    preserve_indentation: bool,
    /// Maximum length for strings (0 for unlimited)
    max_string_length: usize,
    /// Fields to remove based on exact match
    custom_fields_to_remove: HashSet<String>,
    /// Field patterns to remove (e.g., "internal_*")
    custom_field_patterns: HashSet<String>,
    /// Contexts where custom field removal should be skipped
    custom_field_exempt_contexts: HashSet<String>,
    /// Compression level (0-9, where 9 is maximum compression)
    compression_level: u32,
    /// Minimum size for compression (bytes, 0 for always compress)
    compression_threshold: usize,
    /// Whether to compress nested objects independently
    compress_nested: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        let mut enabled_optimizations = HashSet::new();
        enabled_optimizations.insert(OptimizationType::DocumentationRemoval);

        let mut doc_fields = HashSet::new();
        doc_fields.insert("docs".to_string());
        doc_fields.insert("description".to_string());

        let mut preserved_contexts = HashSet::new();
        preserved_contexts.insert("errors".to_string());

        let mut type_identifier_fields = HashSet::new();
        type_identifier_fields.insert("name".to_string());
        type_identifier_fields.insert("type".to_string());
        type_identifier_fields.insert("fields".to_string());

        // Add defaults for string minification
        let mut skip_minification_fields = HashSet::new();
        skip_minification_fields.insert("msg".to_string()); // Don't minify error messages

        let custom_fields_to_remove = HashSet::new();
        let custom_field_patterns = HashSet::new();
        let custom_field_exempt_contexts = HashSet::new();

        Self {
            enabled_optimizations,
            doc_fields,
            preserved_fields: HashSet::new(),
            preserved_contexts,
            type_identifier_fields,
            deduplicate_nested: true,
            skip_minification_fields,
            preserve_indentation: false,
            max_string_length: 0,
            custom_fields_to_remove,
            custom_field_patterns,
            custom_field_exempt_contexts,
            compression_level: 6,  // Default compression level
            compression_threshold: 100,  // Only compress if size > 100 bytes
            compress_nested: false,
        }
    }
}

impl OptimizationConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable a specific optimization
    pub fn enable(&mut self, opt_type: OptimizationType) -> &mut Self {
        self.enabled_optimizations.insert(opt_type);
        self
    }

    /// Disable a specific optimization
    pub fn disable(&mut self, opt_type: OptimizationType) -> &mut Self {
        self.enabled_optimizations.remove(&opt_type);
        self
    }

    /// Add a field to be considered as documentation
    pub fn add_doc_field(&mut self, field: &str) -> &mut Self {
        self.doc_fields.insert(field.to_string());
        self
    }

    /// Add a field to be preserved
    pub fn preserve_field(&mut self, field: &str) -> &mut Self {
        self.preserved_fields.insert(field.to_string());
        self
    }

    /// Add a context where docs should be preserved
    pub fn preserve_context(&mut self, context: &str) -> &mut Self {
        self.preserved_contexts.insert(context.to_string());
        self
    }

    /// Add a field to skip during string minification
    pub fn skip_minification(&mut self, field: &str) -> &mut Self {
        self.skip_minification_fields.insert(field.to_string());
        self
    }

    /// Set maximum string length (0 for unlimited)
    pub fn set_max_string_length(&mut self, length: usize) -> &mut Self {
        self.max_string_length = length;
        self
    }

    /// Set whether to preserve indentation
    pub fn preserve_indentation(&mut self, preserve: bool) -> &mut Self {
        self.preserve_indentation = preserve;
        self
    }

    /// Add a field to be removed
    pub fn remove_field(&mut self, field: &str) -> &mut Self {
        self.custom_fields_to_remove.insert(field.to_string());
        self
    }

    /// Add a field pattern to match for removal (e.g., "internal_*")
    pub fn remove_field_pattern(&mut self, pattern: &str) -> &mut Self {
        self.custom_field_patterns.insert(pattern.to_string());
        self
    }

    /// Add a context where custom field removal should be skipped
    pub fn exempt_context(&mut self, context: &str) -> &mut Self {
        self.custom_field_exempt_contexts.insert(context.to_string());
        self
    }

        /// Set compression level (0-9)
    pub fn set_compression_level(&mut self, level: u32) -> &mut Self {
        self.compression_level = level.min(9);
        self
    }

    /// Set minimum size for compression
    pub fn set_compression_threshold(&mut self, threshold: usize) -> &mut Self {
        self.compression_threshold = threshold;
        self
    }

    /// Set whether to compress nested objects
    pub fn compress_nested(&mut self, enable: bool) -> &mut Self {
        self.compress_nested = enable;
        self
    }
}

/// Tracks metrics for the optimization process
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    /// Original size in bytes
    pub original_size: usize,
    /// Final size after optimization
    pub final_size: usize,
    /// Size reduction per optimization type
    pub size_reduction: HashMap<OptimizationType, usize>,
    /// Processing time per optimization type
    pub processing_time: HashMap<OptimizationType, std::time::Duration>,
    /// Number of items affected per optimization type
    pub items_affected: HashMap<OptimizationType, usize>,
}

impl Default for OptimizationMetrics {
    fn default() -> Self {
        Self {
            original_size: 0,
            final_size: 0,
            size_reduction: HashMap::new(),
            processing_time: HashMap::new(),
            items_affected: HashMap::new(),
        }
    }
}

#[cfg(feature = "parse")]
pub struct IdlOptimizer {
    config: OptimizationConfig,
    metrics: OptimizationMetrics,
    current_context: Vec<String>,
}

#[cfg(feature = "parse")]
impl IdlOptimizer {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            metrics: OptimizationMetrics::default(),
            current_context: Vec::new(),
        }
    }

    pub fn optimize(&mut self, idl: Value) -> Result<(Value, OptimizationMetrics), OptimizeError> {
        let start_time = Instant::now();
        let original_size = serde_json::to_string(&idl)?.len();
        self.metrics.original_size = original_size;

        // Initialize metrics for enabled optimizations
        for opt_type in &self.config.enabled_optimizations {
            self.metrics.items_affected.insert(*opt_type, 0);
            self.metrics.size_reduction.insert(*opt_type, 0);
        }

        let mut current_idl = idl;

        // Apply documentation removal if enabled
        if self.config.enabled_optimizations.contains(&OptimizationType::DocumentationRemoval) {
            let optimization_start = Instant::now();
            current_idl = self.remove_documentation(current_idl)?;
            
            let final_size = serde_json::to_string(&current_idl)?.len();
            let docs_removed = self.metrics.items_affected
                .entry(OptimizationType::DocumentationRemoval)
                .or_default();
            
            self.metrics.processing_time.insert(
                OptimizationType::DocumentationRemoval,
                optimization_start.elapsed()
            );
            self.metrics.size_reduction.insert(
                OptimizationType::DocumentationRemoval,
                original_size - final_size
            );
        }

        // Add type deduplication if enabled
        if self.config.enabled_optimizations.contains(&OptimizationType::TypeDeduplication) {
            let optimization_start = Instant::now();
            
            // Count types before deduplication
            let mut type_map = HashMap::new();
            let mut type_refs = HashMap::new();
            self.catalog_types(&current_idl, &mut type_map, &mut type_refs)?;
            
            let duplicate_count = type_refs.values()
                .filter(|&&count| count > 1)
                .count();
                
            current_idl = self.deduplicate_types(current_idl)?;
            
            let final_size = serde_json::to_string(&current_idl)?.len();
            self.metrics.processing_time.insert(
                OptimizationType::TypeDeduplication,
                optimization_start.elapsed()
            );
            
            // Only insert size reduction if there was actual reduction
            let size_diff = if original_size > final_size {
                original_size - final_size
            } else {
                0
            };
            self.metrics.size_reduction.insert(
                OptimizationType::TypeDeduplication,
                size_diff
            );
            
            self.metrics.items_affected.insert(
                OptimizationType::TypeDeduplication,
                duplicate_count
            );
        }

        // Add string minification if enabled
        if self.config.enabled_optimizations.contains(&OptimizationType::StringMinification) {
            let optimization_start = Instant::now();
            let pre_size = serde_json::to_string(&current_idl)?.len();
            
            current_idl = self.minify_strings(current_idl)?;
            
            let post_size = serde_json::to_string(&current_idl)?.len();
            let minified_count = self.count_minified_strings(&current_idl)?;
            
            self.metrics.processing_time.insert(
                OptimizationType::StringMinification,
                optimization_start.elapsed()
            );
            self.metrics.size_reduction.insert(
                OptimizationType::StringMinification,
                pre_size.saturating_sub(post_size)
            );
        }            

        // Add custom field removal if enabled
        if self.config.enabled_optimizations.contains(&OptimizationType::CustomFieldRemoval) {
            let optimization_start = Instant::now();
            let pre_size = serde_json::to_string(&current_idl)?.len();
            
            // Count fields before removal
            let removed_count = self.count_removed_fields(&current_idl)?;
            
            current_idl = self.remove_custom_fields(current_idl)?;
            
            let post_size = serde_json::to_string(&current_idl)?.len();
            
            self.metrics.processing_time.insert(
                OptimizationType::CustomFieldRemoval,
                optimization_start.elapsed()
            );
            self.metrics.size_reduction.insert(
                OptimizationType::CustomFieldRemoval,
                pre_size.saturating_sub(post_size)
            );
            self.metrics.items_affected.insert(
                OptimizationType::CustomFieldRemoval,
                removed_count
            );
        }

        // Add compression if enabled
        if self.config.enabled_optimizations.contains(&OptimizationType::Compression) {
            let optimization_start = Instant::now();
            let pre_size = serde_json::to_string(&current_idl)?.len();
            
            // Only apply compression if above threshold
            let compressed = self.compress_value(&current_idl)?;
            let compressed_count = self.count_compressed_items(&current_idl)?;
            
            self.metrics.processing_time.insert(
                OptimizationType::Compression,
                optimization_start.elapsed()
            );
            
            // Only create compressed version if data was actually compressed
            if pre_size > self.config.compression_threshold {
                let mut compressed_map = Map::new();
                compressed_map.insert(
                    "compressed_data".to_string(),
                    Value::String(STANDARD.encode(&compressed))  // Just use encode directly
                );
                compressed_map.insert(
                    "original_size".to_string(),
                    Value::Number(pre_size.into())
                );
                current_idl = Value::Object(compressed_map);
                
                self.metrics.size_reduction.insert(
                    OptimizationType::Compression,
                    pre_size.saturating_sub(compressed.len())
                );
            }
            
            self.metrics.items_affected.insert(
                OptimizationType::Compression,
                compressed_count
            );
        }

        self.metrics.final_size = serde_json::to_string(&current_idl)?.len();
        
        Ok((current_idl, self.metrics.clone()))
    }

    fn remove_documentation(&mut self, value: Value) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => self.process_object(map),
            Value::Array(arr) => {
                let processed: Result<Vec<_>, _> = arr
                    .into_iter()
                    .map(|v| self.remove_documentation(v))
                    .collect();
                Ok(Value::Array(processed?))
            }
            _ => Ok(value),
        }
    }

    fn process_object(&mut self, map: Map<String, Value>) -> Result<Value, OptimizeError> {
        let mut new_map = Map::new();

        // Check if we're in a preserved context
        let should_preserve = self.current_context
            .last()
            .map(|ctx| self.config.preserved_contexts.contains(ctx))
            .unwrap_or(false);

        for (key, value) in map {
            // Track context for nested processing
            if value.is_object() || value.is_array() {
                self.current_context.push(key.clone());
            }

            let should_remove = self.config.doc_fields.contains(&key) && 
                              !self.config.preserved_fields.contains(&key) &&
                              !should_preserve;

            if should_remove {
                *self.metrics.items_affected
                    .entry(OptimizationType::DocumentationRemoval)
                    .or_insert(0) += 1;
            } else {
                let processed_value = self.remove_documentation(value)?;
                new_map.insert(key.clone(), processed_value);
            }

            // Pop context after processing
            if self.current_context.last().map(|s| s == &key).unwrap_or(false) {
                self.current_context.pop();
            }
        }

        Ok(Value::Object(new_map))
    }

    fn deduplicate_types(&mut self, value: Value) -> Result<Value, OptimizeError> {
        // Track unique types we've seen
        let mut type_map: HashMap<String, Value> = HashMap::new();
        let mut type_references: HashMap<String, usize> = HashMap::new();
        
        // First pass: identify and catalog unique types
        self.catalog_types(&value, &mut type_map, &mut type_references)?;
        
        // Second pass: replace duplicate types with references
        self.replace_duplicate_types(value, &type_map, &mut type_references)
    }

    fn catalog_types(
        &self,
        value: &Value,
        type_map: &mut HashMap<String, Value>,
        type_references: &mut HashMap<String, usize>,
    ) -> Result<(), OptimizeError> {
        match value {
            Value::Object(map) => {
                // Check if this object is a type definition
                if self.is_type_definition(map) {
                    let type_hash = self.compute_type_hash(map)?;
                    
                    if type_map.contains_key(&type_hash) {
                        // Increment reference count for existing type
                        *type_references.entry(type_hash).or_insert(0) += 1;
                    } else {
                        // Add new type and initialize reference count
                        type_map.insert(type_hash.clone(), Value::Object(map.clone()));
                        type_references.insert(type_hash, 1);
                    }
                }
    
                // Process nested objects
                for (_, v) in map {
                    self.catalog_types(v, type_map, type_references)?;
                }
            }
            Value::Array(arr) => {
                for v in arr {
                    self.catalog_types(v, type_map, type_references)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn is_type_definition(&self, map: &Map<String, Value>) -> bool {
        // Must have type field
        let has_type = map.get("type")
            .and_then(|v| v.as_str())
            .map_or(false, |t| t == "struct");
            
        // Must have fields array
        let has_fields = map.get("fields")
            .and_then(|v| v.as_array())
            .map_or(false, |fields| !fields.is_empty());
            
        has_type && has_fields
    }

    fn compute_type_hash(&self, map: &Map<String, Value>) -> Result<String, OptimizeError> {
        let mut hash_input = String::new();
        
        // Add type (should be "struct")
        if let Some(type_val) = map.get("type").and_then(|v| v.as_str()) {
            hash_input.push_str(type_val);
        }
        
        // Add fields in a deterministic order
        if let Some(fields) = map.get("fields").and_then(|v| v.as_array()) {
            let mut field_strings = Vec::new();
            
            for field in fields {
                if let Some(field_obj) = field.as_object() {
                    let field_type = field_obj.get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                        
                    field_strings.push(format!("{}:{}", 
                        field_obj.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        field_type
                    ));
                }
            }
            
            // Sort for consistent ordering
            field_strings.sort();
            hash_input.push_str(&field_strings.join(","));
        }
        
        Ok(format!("{:x}", compute(hash_input.as_bytes())))
    }

    fn count_types(&self, value: &Value) -> Result<usize, OptimizeError> {
        let mut count = 0;
        
        match value {
            Value::Object(map) => {
                if self.is_type_definition(map) {
                    count += 1;
                }
                for (_, v) in map {
                    count += self.count_types(v)?;
                }
            }
            Value::Array(arr) => {
                for v in arr {
                    count += self.count_types(v)?;
                }
            }
            _ => {}
        }
        
        Ok(count)
    }

    fn replace_duplicate_types(
        &mut self,
        value: Value,
        type_map: &HashMap<String, Value>,
        type_references: &mut HashMap<String, usize>,
    ) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                if self.is_type_definition(&map) {
                    let type_hash = self.compute_type_hash(&map)?;
                    
                    // Replace with reference only if it's a duplicate and not the first occurrence
                    if type_references.get(&type_hash).map_or(false, |&count| count > 1) {
                        if let Some(first_type) = type_map.get(&type_hash) {
                            // Only replace if this isn't the first occurrence
                            if !Value::Object(map.clone()).eq(first_type) {
                                let mut reference_map = Map::new();
                                reference_map.insert("type_ref".to_string(), Value::String(type_hash));
                                return Ok(Value::Object(reference_map));
                            }
                        }
                    }
                }
    
                // Process nested objects
                let mut new_map = Map::new();
                for (k, v) in map {
                    new_map.insert(
                        k,
                        self.replace_duplicate_types(v, type_map, type_references)?
                    );
                }
                Ok(Value::Object(new_map))
            }
            Value::Array(arr) => {
                let processed: Result<Vec<_>, _> = arr
                    .into_iter()
                    .map(|v| self.replace_duplicate_types(v, type_map, type_references))
                    .collect();
                Ok(Value::Array(processed?))
            }
            _ => Ok(value),
        }
    }

    fn minify_strings(&mut self, value: Value) -> Result<Value, OptimizeError> {
        match value {
            Value::String(s) => {
                let original = s.clone();
                let minified = self.minify_string(&s)?;
                
                if minified != original {
                    *self.metrics.items_affected
                        .entry(OptimizationType::StringMinification)
                        .or_insert(0) += 1;
                }
                
                Ok(Value::String(minified))
            }
            Value::Object(map) => {
                let mut new_map = Map::new();
                for (key, val) in map {
                    let processed_value = if self.config.skip_minification_fields.contains(&key) {
                        val
                    } else {
                        self.minify_strings(val)?
                    };
                    new_map.insert(key, processed_value);
                }
                Ok(Value::Object(new_map))
            }
            Value::Array(arr) => {
                let processed: Result<Vec<_>, _> = arr
                    .into_iter()
                    .map(|v| self.minify_strings(v))
                    .collect();
                Ok(Value::Array(processed?))
            }
            _ => Ok(value),
        }
    }

    fn minify_string(&self, input: &str) -> Result<String, OptimizeError> {
        
        // First, normalize the whitespace
        let normalized = if self.config.preserve_indentation {
            input.lines()
                .map(|line| line.trim_end())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            input.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        };
    
        // Handle max length
        if self.config.max_string_length > 0 && normalized.chars().count() > self.config.max_string_length {
            let mut result: String = normalized
                .chars()
                .take(self.config.max_string_length - 3)
                .collect();
            result.push_str("...");
            Ok(result)
        } else {
            Ok(normalized)
        }
    }
    

    fn count_minified_strings(&self, value: &Value) -> Result<usize, OptimizeError> {
        match value {
            Value::String(s) => {
                // Check if string would be modified
                let normalized = if self.config.preserve_indentation {
                    s.lines()
                        .map(|line| line.trim_end())
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    s.split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                };
    
                let would_be_minified = s != &normalized;
                let would_be_truncated = self.config.max_string_length > 0 
                    && s.chars().count() > self.config.max_string_length;
    
                Ok(if would_be_minified || would_be_truncated { 1 } else { 0 })
            },
            Value::Object(map) => {
                let mut count = 0;
                for (key, val) in map {
                    if !self.config.skip_minification_fields.contains(key) {
                        count += self.count_minified_strings(val)?;
                    }
                }
                Ok(count)
            }
            Value::Array(arr) => {
                let mut count = 0;
                for val in arr {
                    count += self.count_minified_strings(val)?;
                }
                Ok(count)
            }
            _ => Ok(0),
        }
    }

    fn remove_custom_fields(&mut self, value: Value) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                let mut new_map = Map::new();
                
                // Check if we're in an exempt context
                let should_exempt = self.current_context
                    .last()
                    .map(|ctx| self.config.custom_field_exempt_contexts.contains(ctx))
                    .unwrap_or(false);

                for (key, value) in map {
                    // Track context for nested processing
                    if value.is_object() || value.is_array() {
                        self.current_context.push(key.clone());
                    }

                    let should_remove = !should_exempt && 
                        (self.config.custom_fields_to_remove.contains(&key) ||
                         self.matches_pattern(&key));

                    if !should_remove {
                        let processed_value = self.remove_custom_fields(value)?;
                        new_map.insert(key.clone(), processed_value);
                    } else {
                        *self.metrics.items_affected
                            .entry(OptimizationType::CustomFieldRemoval)
                            .or_insert(0) += 1;
                    }

                    // Pop context after processing
                    if self.current_context.last().map(|s| s == &key).unwrap_or(false) {
                        self.current_context.pop();
                    }
                }

                Ok(Value::Object(new_map))
            }
            Value::Array(arr) => {
                let processed: Result<Vec<_>, _> = arr
                    .into_iter()
                    .map(|v| self.remove_custom_fields(v))
                    .collect();
                Ok(Value::Array(processed?))
            }
            _ => Ok(value),
        }
    }

    fn matches_pattern(&self, field: &str) -> bool {
        for pattern in &self.config.custom_field_patterns {
            if pattern.ends_with('*') {
                let prefix = &pattern[..pattern.len() - 1];
                if field.starts_with(prefix) {
                    return true;
                }
            } else {
                if field == pattern {
                    return true;
                }
            }
        }
        false
    }

    fn count_removed_fields(&mut self, value: &Value) -> Result<usize, OptimizeError> {
        match value {
            Value::Object(map) => {
                let mut count = 0;
                
                // Check if current context is exempt
                let should_exempt = self.current_context
                    .last()
                    .map(|ctx| self.config.custom_field_exempt_contexts.contains(ctx))
                    .unwrap_or(false);
    
                if !should_exempt {
                    for (key, val) in map {
                        if self.config.custom_fields_to_remove.contains(key) || 
                           self.matches_pattern(key) {
                            count += 1;
                        } else if val.is_object() || val.is_array() {
                            // Track context for nested objects
                            self.current_context.push(key.clone());
                            count += self.count_removed_fields(val)?;
                            self.current_context.pop();
                        }
                    }
                }
    
                Ok(count)
            }
            Value::Array(arr) => {
                let mut count = 0;
                for val in arr {
                    count += self.count_removed_fields(val)?;
                }
                Ok(count)
            }
            _ => Ok(0),
        }
    }

    fn compress_value(&self, value: &Value) -> Result<Vec<u8>, OptimizeError> {
        let json_str = serde_json::to_string(value)?;
        
        // Return uncompressed data if below threshold
        if json_str.len() <= self.config.compression_threshold {
            return Ok(json_str.into_bytes());
        }
    
        let mut encoder = flate2::write::ZlibEncoder::new(
            Vec::new(),
            flate2::Compression::new(self.config.compression_level)
        );
        encoder.write_all(json_str.as_bytes())?;
        Ok(encoder.finish()?)
    }

    fn count_compressed_items(&self, value: &Value) -> Result<usize, OptimizeError> {
        match value {
            Value::Object(map) => {
                let mut count = 0;
                let json_str = serde_json::to_string(value)?;
                
                // Count this object if it exceeds threshold
                if json_str.len() > self.config.compression_threshold {
                    count += 1;
                }

                // Count nested objects if enabled
                if self.config.compress_nested {
                    for (_, v) in map {
                        count += self.count_compressed_items(v)?;
                    }
                }
                
                Ok(count)
            }
            Value::Array(arr) => {
                let mut count = 0;
                if self.config.compress_nested {
                    for v in arr {
                        count += self.count_compressed_items(v)?;
                    }
                }
                Ok(count)
            }
            _ => Ok(0),
        }
    }
}

#[derive(Debug)]
pub enum OptimizeError {
    SerdeError(serde_json::Error),
    InvalidValue(String),
    CompressionError(std::io::Error),
}

impl From<serde_json::Error> for OptimizeError {
    fn from(err: serde_json::Error) -> Self {
        OptimizeError::SerdeError(err)
    }
}

impl From<std::io::Error> for OptimizeError {
    fn from(err: std::io::Error) -> Self {
        OptimizeError::CompressionError(err)
    }
}