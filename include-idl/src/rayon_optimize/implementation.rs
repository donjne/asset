#[cfg(feature = "parse-rayon")]
use serde_json::{Map, Value};
use std::{collections::{HashMap, HashSet}, time::Instant};
use std::sync::{Arc, Mutex};
use md5::compute;
use std::io::Write;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use rayon::prelude::*;

/// Types of optimizations available
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum OptimizationType {
    DocumentationRemoval,
    TypeDeduplication,
    StringMinification,
    CustomFieldRemoval,
    Compression,
}

struct CompressionAnalysis {
    original_size: usize,
    estimated_reduction_bps: u32,  // basis points (1/100th of a percent)
    compression_ratio_bps: u32,    // basis points
    recommended_level: u32,
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

        // Defaults for string minification
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

    /// Field to be considered as documentation
    pub fn add_doc_field(&mut self, field: &str) -> &mut Self {
        self.doc_fields.insert(field.to_string());
        self
    }

    /// Field to be preserved
    pub fn preserve_field(&mut self, field: &str) -> &mut Self {
        self.preserved_fields.insert(field.to_string());
        self
    }

    /// Context where docs should be preserved
    pub fn preserve_context(&mut self, context: &str) -> &mut Self {
        self.preserved_contexts.insert(context.to_string());
        self
    }

    /// Field to skip during string minification
    pub fn skip_minification(&mut self, field: &str) -> &mut Self {
        self.skip_minification_fields.insert(field.to_string());
        self
    }

    /// Set the maximum string length (0 for unlimited)
    pub fn set_max_string_length(&mut self, length: usize) -> &mut Self {
        self.max_string_length = length;
        self
    }

    /// Set whether to preserve indentation
    pub fn preserve_indentation(&mut self, preserve: bool) -> &mut Self {
        self.preserve_indentation = preserve;
        self
    }

    /// Field to be removed
    pub fn remove_field(&mut self, field: &str) -> &mut Self {
        self.custom_fields_to_remove.insert(field.to_string());
        self
    }

    /// Field pattern to match for removal (e.g., "internal_*")
    pub fn remove_field_pattern(&mut self, pattern: &str) -> &mut Self {
        self.custom_field_patterns.insert(pattern.to_string());
        self
    }

    /// Context where custom field removal should be skipped
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

struct TypeCache {
    hash_map: Arc<Mutex<HashMap<String, String>>>,  // Cache for computed hashes
    type_map: Arc<Mutex<HashMap<String, Value>>>,   // Cache for type definitions
}

impl TypeCache {
    fn new() -> Self {
        Self {
            hash_map: Arc::new(Mutex::new(HashMap::new())),
            type_map: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn get_or_compute_hash(&self, map: &Map<String, Value>, _config: &OptimizationConfig) -> Result<String, OptimizeError> {
        let key = self.make_cache_key(map);
        
        // Try to get from hash_map first
        {
            let hash_map = self.hash_map.lock().map_err(|e| OptimizeError::ThreadError(e.to_string()))?;
            if let Some(hash) = hash_map.get(&key) {
                return Ok(hash.clone());
            }
        }
        
        // If not found, compute and insert
        let hash = Self::compute_type_hash(map)?;
        {
            let mut hash_map = self.hash_map.lock().map_err(|e| OptimizeError::ThreadError(e.to_string()))?;
            hash_map.insert(key, hash.clone());
        }
        Ok(hash)
    }

    fn make_cache_key(&self, map: &Map<String, Value>) -> String {
        let mut fields: Vec<(String, String)> = map.iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect();
        fields.sort();
        fields.iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn compute_type_hash(map: &Map<String, Value>) -> Result<String, OptimizeError> {
        let mut hash_input = String::new();
        
        if let Some(type_val) = map.get("type").and_then(|v| v.as_str()) {
            hash_input.push_str(type_val);
        }
        
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
            
            field_strings.sort();
            hash_input.push_str(&field_strings.join(","));
        }
        
        Ok(format!("{:x}", compute(hash_input.as_bytes())))
    }
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
        let _start_time = Instant::now();
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
            let _docs_removed = self.metrics.items_affected
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
            
        // Create TypeCache first
        let type_cache = TypeCache::new();
        let type_refs = Arc::new(Mutex::new(HashMap::new()));

        // Use type_cache 
        self.catalog_types_parallel(&current_idl, &type_cache, &type_refs)?;
        
        // Lock the mutex to access values
        let duplicate_count = {
            let refs = type_refs.lock()
                .map_err(|e| OptimizeError::ThreadError(e.to_string()))?;
            refs.values()
                .filter(|&&count| count > 1)
                .count()
        };
                
        // Pass type_cache to replace_duplicate_types
        current_idl = self.replace_duplicate_types_parallel(current_idl, &type_cache, &type_refs)?;
            
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
            let _minified_count = self.count_minified_strings(&current_idl)?;
            
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

        // Apply compression if enabled
        if self.config.enabled_optimizations.contains(&OptimizationType::Compression) {
            let optimization_start = Instant::now();
            let pre_size = serde_json::to_string(&current_idl)?.len();
            
            let mut compressed_count = 0;
            let mut total_size_reduction = 0;

            // Always analyze first
            let analysis = self.analyze_compression_potential(&serde_json::to_string(&current_idl)?)?;
            // println!("Compression analysis results:");
            // println!("  Original size: {}", analysis.original_size);
            // println!("  Estimated reduction (bps): {}", analysis.estimated_reduction_bps);
            // println!("  Compression ratio (bps): {}", analysis.compression_ratio_bps);
            // println!("  Recommended level: {}", analysis.recommended_level);
            
            if self.should_compress(&analysis) {
                if self.config.compress_nested {
                    // Use our static version 
                    let config = Arc::new(self.config.clone());
                    let metrics = Arc::new(Mutex::new(&mut self.metrics));
                    let pre_compression_size = serde_json::to_string(&current_idl)?.len();
                    
                    current_idl = Self::compress_nested_parallel_static(
                        &current_idl,
                        &[],
                        config,
                        metrics.clone(),
                        pre_compression_size
                    )?;
                    
                    // Get compression count (in a separate scope)
                    {
                        let guard = metrics.lock()
                            .map_err(|e| OptimizeError::ThreadError(e.to_string()))?;
                        compressed_count = *guard.items_affected
                            .get(&OptimizationType::Compression)
                            .unwrap_or(&0);
                    }
                    
                    // Calculate size reduction
                    let post_compression_size = serde_json::to_string(&current_idl)?.len();
                    total_size_reduction = pre_compression_size.saturating_sub(post_compression_size);
                } else {
                    // Single compression
                    let compressed = self.compress_value(&current_idl)?;
                    compressed_count = 1;
                    
                    let mut compressed_map = Map::new();
                    compressed_map.insert(
                        "compressed_data".to_string(),
                        Value::String(STANDARD.encode(&compressed))
                    );
                    compressed_map.insert(
                        "original_size".to_string(),
                        Value::Number(pre_size.into())
                    );
                    current_idl = Value::Object(compressed_map);
                    
                    total_size_reduction = pre_size.saturating_sub(compressed.len());
                }
            } else {
                println!("Compression criteria not met, skipping compression");
            }
            
            // Update metrics
            self.metrics.size_reduction.insert(
                OptimizationType::Compression,
                total_size_reduction
            );
            self.metrics.items_affected.insert(
                OptimizationType::Compression,
                compressed_count
            );
            self.metrics.processing_time.insert(
                OptimizationType::Compression,
                optimization_start.elapsed()
            );
        }
    

        self.metrics.final_size = serde_json::to_string(&current_idl)?.len();
        
        Ok((current_idl, self.metrics.clone()))
    }

    fn should_remove_doc_field(&self, key: &str, context: &[String]) -> bool {
        // Quick pre-check for documentation fields
        if !self.config.doc_fields.contains(key) {
            return false;
        }

        // Check preservation rules
        if self.config.preserved_fields.contains(key) {
            return false;
        }

        // Check context preservation
        if let Some(ctx) = context.last() {
            if self.config.preserved_contexts.contains(ctx) {
                return false;
            }
        }

        true
    }

    fn remove_documentation(&mut self, value: Value) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                let items: Vec<_> = map.into_iter().collect();
                let config = Arc::new(self.config.clone());
                let metrics = Arc::new(Mutex::new(&mut self.metrics));
                let context = Arc::new(self.current_context.clone());

                if items.len() > 20 {  // Threshold for parallel processing
                    let processed: Result<Vec<_>, OptimizeError> = items
                        .into_par_iter()
                        .filter_map(|(key, value)| {
                            let should_remove = Self::should_remove_doc_field_static(
                                &key,
                                &context,
                                &config
                            );

                            if should_remove {
                                if let Ok(mut metrics) = metrics.lock() {
                                    *metrics.items_affected
                                        .entry(OptimizationType::DocumentationRemoval)
                                        .or_insert(0) += 1;
                                }
                                None
                            } else {
                                Some((key, value))
                            }
                        })
                        .map(|(key, value)| {
                            let mut new_context = (*context).clone();
                            if value.is_object() || value.is_array() {
                                new_context.push(key.clone());
                            }
                            let result = Self::process_value_static(
                                value,
                                &new_context,
                                config.clone(),
                                metrics.clone()
                            )?;
                            Ok((key, result))
                        })
                        .collect();

                    let mut new_map = Map::new();
                    for (key, value) in processed? {
                        new_map.insert(key, value);
                    }
                    Ok(Value::Object(new_map))
                } else {
                    // Sequential processing for smaller objects
                    let mut new_map = Map::with_capacity(items.len());
                    for (key, value) in items {
                        if !self.should_remove_doc_field(&key, &self.current_context) {
                            let processed_value = if value.is_object() || value.is_array() {
                                self.current_context.push(key.clone());
                                let result = self.remove_documentation(value)?;
                                self.current_context.pop();
                                result
                            } else {
                                value
                            };
                            new_map.insert(key, processed_value);
                        } else {
                            *self.metrics.items_affected
                                .entry(OptimizationType::DocumentationRemoval)
                                .or_insert(0) += 1;
                        }
                    }
                    Ok(Value::Object(new_map))
                }
            }
            Value::Array(arr) => {
                if arr.len() > 20 {
                    let config = Arc::new(self.config.clone());
                    let metrics = Arc::new(Mutex::new(&mut self.metrics));
                    let context = Arc::new(self.current_context.clone());

                    let processed: Result<Vec<_>, _> = arr
                        .into_par_iter()
                        .map(|v| {
                            Self::process_value_static(
                                v,
                                &context,
                                config.clone(),
                                metrics.clone()
                            )
                        })
                        .collect();
                    Ok(Value::Array(processed?))
                } else {
                    let processed: Result<Vec<_>, _> = arr
                        .into_iter()
                        .map(|v| self.remove_documentation(v))
                        .collect();
                    Ok(Value::Array(processed?))
                }
            }
            _ => Ok(value),
        }
    }

    fn should_remove_doc_field_static(
        key: &str,
        context: &[String],
        config: &OptimizationConfig
    ) -> bool {
        if !config.doc_fields.contains(key) {
            return false;
        }

        if config.preserved_fields.contains(key) {
            return false;
        }

        if let Some(ctx) = context.last() {
            if config.preserved_contexts.contains(ctx) {
                return false;
            }
        }

        true
    }

    fn process_value_static(
        value: Value,
        context: &[String],
        config: Arc<OptimizationConfig>,
        metrics: Arc<Mutex<&mut OptimizationMetrics>>,
    ) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                let items: Vec<_> = map.into_iter().collect();
                let mut new_map = Map::with_capacity(items.len());

                for (key, val) in items {
                    if !Self::should_remove_doc_field_static(&key, context, &config) {
                        let mut new_context = context.to_vec();
                        if val.is_object() || val.is_array() {
                            new_context.push(key.clone());
                        }
                        let processed_value = Self::process_value_static(
                            val,
                            &new_context,
                            config.clone(),
                            metrics.clone()
                        )?;
                        new_map.insert(key, processed_value);
                    } else {
                        if let Ok(mut metrics) = metrics.lock() {
                            *metrics.items_affected
                                .entry(OptimizationType::DocumentationRemoval)
                                .or_insert(0) += 1;
                        }
                    }
                }
                Ok(Value::Object(new_map))
            }
            Value::Array(arr) => {
                let processed: Result<Vec<_>, _> = arr
                    .into_iter()
                    .map(|v| Self::process_value_static(
                        v,
                        context,
                        config.clone(),
                        metrics.clone()
                    ))
                    .collect();
                Ok(Value::Array(processed?))
            }
            _ => Ok(value),
        }
    }

    fn replace_duplicate_types_parallel(
        &self,
        value: Value,
        type_cache: &TypeCache,
        type_references:  &Arc<Mutex<HashMap<String, usize>>>,
    ) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                if self.is_type_definition(&map) {
                    let type_hash = type_cache.get_or_compute_hash(&map, &self.config)?;
                    
                    let refs_count = {
                        let refs = type_references.lock()
                            .map_err(|e| OptimizeError::ThreadError(e.to_string()))?;
                        refs.get(&type_hash).cloned().unwrap_or(0)
                    };
                    
                    if refs_count > 1 {
                        let type_map = type_cache.type_map.lock()
                            .map_err(|e| OptimizeError::ThreadError(e.to_string()))?;
                            
                        if let Some(first_type) = type_map.get(&type_hash) {
                            if !Value::Object(map.clone()).eq(first_type) {
                                let mut reference_map = Map::new();
                                reference_map.insert("type_ref".to_string(), Value::String(type_hash));
                                return Ok(Value::Object(reference_map));
                            }
                        }
                    }
                }

                let items: Vec<_> = map.into_iter().collect();
                if items.len() > 20 {
                    let processed: Result<Vec<(String, Value)>, OptimizeError> = items
                        .into_par_iter()
                        .map(|(k, v)| -> Result<(String, Value), OptimizeError> {
                            Ok((k, self.replace_duplicate_types_parallel(v, type_cache, type_references)?))
                        })
                        .collect();

                    let mut new_map = Map::new();
                    for (k, v) in processed? {
                        new_map.insert(k, v);
                    }
                    Ok(Value::Object(new_map))
                } else {
                    let mut new_map = Map::new();
                    for (k, v) in items {
                        new_map.insert(k, self.replace_duplicate_types_parallel(v, type_cache, type_references)?);
                    }
                    Ok(Value::Object(new_map))
                }
            }
            Value::Array(arr) => {
                if arr.len() > 20 {
                    let processed: Result<Vec<Value>, OptimizeError> = arr
                        .into_par_iter()
                        .map(|v| self.replace_duplicate_types_parallel(v, type_cache, type_references))
                        .collect();
                    Ok(Value::Array(processed?))
                } else {
                    let processed: Result<Vec<Value>, OptimizeError> = arr
                        .into_iter()
                        .map(|v| self.replace_duplicate_types_parallel(v, type_cache, type_references))
                        .collect();
                    Ok(Value::Array(processed?))
                }
            }
            _ => Ok(value),
        }
    }

    fn catalog_types_parallel(
        &self,
        value: &Value,
        type_cache: &TypeCache,
        type_references:  &Arc<Mutex<HashMap<String, usize>>>,
    ) -> Result<(), OptimizeError> {
        match value {
            Value::Object(map) => {
                // First check if this is a type definition and process it
                if self.is_type_definition(map) {
                    let type_hash = type_cache.get_or_compute_hash(map, &self.config)?;
                    
                    {
                        let mut type_map = type_cache.type_map.lock()
                            .map_err(|e| OptimizeError::ThreadError(e.to_string()))?;
                        
                        if !type_map.contains_key(&type_hash) {
                            type_map.insert(type_hash.clone(), Value::Object(map.clone()));
                        }
                    }
                    
                    {
                        let mut refs = type_references.lock()
                            .map_err(|e| OptimizeError::ThreadError(e.to_string()))?;
                        *refs.entry(type_hash).or_insert(0) += 1;
                    }
                }
    
                // Then process nested objects in parallel if there are enough of them
                let values: Vec<_> = map.values().collect();
                if values.len() > 20 {
                    let type_cache = Arc::new(Mutex::new(type_cache));
                    let type_refs = Arc::new(Mutex::new(type_references));
    
                    values.par_iter().try_for_each(|v| -> Result<(), OptimizeError> {
                        let mut cache_guard = type_cache.lock().map_err(|e| 
                            OptimizeError::ThreadError(e.to_string()))?;
                        let mut refs_guard = type_refs.lock().map_err(|e| 
                            OptimizeError::ThreadError(e.to_string()))?;
                        
                        self.catalog_types_parallel(v, &mut cache_guard, &mut refs_guard)
                    })?;
                } else {
                    // Sequential processing for smaller objects
                    for v in map.values() {
                        self.catalog_types_parallel(v, type_cache, type_references)?;
                    }
                }
            }
            Value::Array(arr) => {
                if arr.len() > 20 {
                    let type_cache = Arc::new(Mutex::new(type_cache));
                    let type_refs = Arc::new(Mutex::new(type_references));
    
                    arr.par_iter().try_for_each(|v| -> Result<(), OptimizeError> {
                        let mut cache_guard = type_cache.lock().map_err(|e| 
                            OptimizeError::ThreadError(e.to_string()))?;
                        let mut refs_guard = type_refs.lock().map_err(|e| 
                            OptimizeError::ThreadError(e.to_string()))?;
                        
                        self.catalog_types_parallel(v, &mut cache_guard, &mut refs_guard)
                    })?;
                } else {
                    for v in arr {
                        self.catalog_types_parallel(v, type_cache, type_references)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn is_type_definition(&self, map: &Map<String, Value>) -> bool {
        // Check all configured type identifier fields
        for field in &self.config.type_identifier_fields {
            if !map.contains_key(field) {
                return false;
            }
        }
        
        // Check specific struct requirements
        let has_type = map.get("type")
            .and_then(|v| v.as_str())
            .map_or(false, |t| t == "struct");
            
        let has_fields = map.get("fields")
            .and_then(|v| v.as_array())
            .map_or(false, |fields| !fields.is_empty());
            
        has_type && has_fields
    }

    fn minify_strings(&mut self, value: Value) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                let items: Vec<_> = map.into_iter().collect();
                let config = Arc::new(self.config.clone());
                let metrics = Arc::new(Mutex::new(&mut self.metrics));
    
                if items.len() > 20 {
                    let processed: Result<Vec<_>, OptimizeError> = items
                        .into_par_iter()
                        .map(|(key, val)| -> Result<(String, Value), OptimizeError> {
                            let processed_value = if config.skip_minification_fields.contains(&key) {
                                val
                            } else {
                                Self::minify_value_static(
                                    val.clone(),
                                    &config,
                                    metrics.clone()
                                )?
                            };
                            Ok((key, processed_value))
                        })
                        .collect();
    
                    let mut new_map = Map::new();
                    for (key, value) in processed? {
                        new_map.insert(key, value);
                    }
                    Ok(Value::Object(new_map))
                } else {
                    let mut new_map = Map::new();
                    for (key, val) in items {
                        let processed_value = if self.config.skip_minification_fields.contains(&key) {
                            val
                        } else {
                            self.minify_strings(val)?
                        };
                        new_map.insert(key, processed_value);
                    }
                    Ok(Value::Object(new_map))
                }
            }
            Value::Array(arr) => {
                if arr.len() > 20 {
                    let config = Arc::new(self.config.clone());
                    let metrics = Arc::new(Mutex::new(&mut self.metrics));
    
                    let processed: Result<Vec<_>, OptimizeError> = arr
                        .into_par_iter()
                        .map(|v| Self::minify_value_static(v, &config, metrics.clone()))
                        .collect();
                    Ok(Value::Array(processed?))
                } else {
                    let processed: Result<Vec<_>, OptimizeError> = arr
                        .into_iter()
                        .map(|v| self.minify_strings(v))
                        .collect();
                    Ok(Value::Array(processed?))
                }
            }
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
            _ => Ok(value),
        }
    }

    fn minify_string(&self, input: &str) -> Result<String, OptimizeError> {
        
        // Normalize the whitespace first
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
    
        // Then handle max length
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

    fn minify_value_static(
        value: Value,
        config: &OptimizationConfig,
        metrics: Arc<Mutex<&mut OptimizationMetrics>>,
    ) -> Result<Value, OptimizeError> {
        match value {
            Value::String(s) => {
                let original = s.clone();
                let normalized = if config.preserve_indentation {
                    s.lines()
                        .map(|line| line.trim_end())
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    s.split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                };
    
                let minified = if config.max_string_length > 0 
                    && normalized.chars().count() > config.max_string_length {
                    let mut result: String = normalized
                        .chars()
                        .take(config.max_string_length - 3)
                        .collect();
                    result.push_str("...");
                    result
                } else {
                    normalized
                };
    
                if minified != original {
                    if let Ok(mut metrics) = metrics.lock() {
                        *metrics.items_affected
                            .entry(OptimizationType::StringMinification)
                            .or_insert(0) += 1;
                    }
                }
    
                Ok(Value::String(minified))
            }
            Value::Object(map) => {
                let mut new_map = Map::new();
                for (key, val) in map {
                    let processed_value = if config.skip_minification_fields.contains(&key) {
                        val
                    } else {
                        Self::minify_value_static(val, config, metrics.clone())?
                    };
                    new_map.insert(key, processed_value);
                }
                Ok(Value::Object(new_map))
            }
            Value::Array(arr) => {
                let processed: Result<Vec<_>, OptimizeError> = arr
                    .into_iter()
                    .map(|v| Self::minify_value_static(v, config, metrics.clone()))
                    .collect();
                Ok(Value::Array(processed?))
            }
            _ => Ok(value),
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
                let items: Vec<_> = map.into_iter().collect();
                let config = Arc::new(self.config.clone());
                let metrics = Arc::new(Mutex::new(&mut self.metrics));
                let context = Arc::new(self.current_context.clone());
    
                if items.len() > 20 {  // Still the same threshold as other previous optimizations
                    let processed: Result<Vec<_>, OptimizeError> = items
                        .into_par_iter()
                        .filter_map(|(key, value)| {
                            let should_remove = Self::should_remove_custom_field_static(
                                &key,
                                &context,
                                &config
                            );
    
                            if should_remove {
                                if let Ok(mut metrics) = metrics.lock() {
                                    *metrics.items_affected
                                        .entry(OptimizationType::CustomFieldRemoval)
                                        .or_insert(0) += 1;
                                }
                                None
                            } else {
                                Some((key, value))
                            }
                        })
                        .map(|(key, value)| {
                            let mut new_context = (*context).clone();
                            if value.is_object() || value.is_array() {
                                new_context.push(key.clone());
                            }
                            let result = Self::process_custom_fields_static(
                                value,
                                &new_context,
                                config.clone(),
                                metrics.clone()
                            )?;
                            Ok((key, result))
                        })
                        .collect();
    
                    let mut new_map = Map::new();
                    for (key, value) in processed? {
                        new_map.insert(key, value);
                    }
                    Ok(Value::Object(new_map))
                } else {
                    // Sequential processing for smaller objects
                    let mut new_map = Map::with_capacity(items.len());
                    for (key, value) in items {
                        let should_exempt = self.current_context
                            .last()
                            .map(|ctx| self.config.custom_field_exempt_contexts.contains(ctx))
                            .unwrap_or(false);
    
                        let should_remove = !should_exempt && 
                            (self.config.custom_fields_to_remove.contains(&key) ||
                             self.matches_pattern(&key));
    
                        if !should_remove {
                            let processed_value = if value.is_object() || value.is_array() {
                                self.current_context.push(key.clone());
                                let result = self.remove_custom_fields(value)?;
                                self.current_context.pop();
                                result
                            } else {
                                value
                            };
                            new_map.insert(key, processed_value);
                        } else {
                            *self.metrics.items_affected
                                .entry(OptimizationType::CustomFieldRemoval)
                                .or_insert(0) += 1;
                        }
                    }
                    Ok(Value::Object(new_map))
                }
            }
            Value::Array(arr) => {
                if arr.len() > 20 {
                    let config = Arc::new(self.config.clone());
                    let metrics = Arc::new(Mutex::new(&mut self.metrics));
                    let context = Arc::new(self.current_context.clone());
    
                    let processed: Result<Vec<_>, _> = arr
                        .into_par_iter()
                        .map(|v| {
                            Self::process_custom_fields_static(
                                v,
                                &context,
                                config.clone(),
                                metrics.clone()
                            )
                        })
                        .collect();
                    Ok(Value::Array(processed?))
                } else {
                    let processed: Result<Vec<_>, _> = arr
                        .into_iter()
                        .map(|v| self.remove_custom_fields(v))
                        .collect();
                    Ok(Value::Array(processed?))
                }
            }
            _ => Ok(value),
        }
    }


    fn should_remove_custom_field_static(
        key: &str,
        context: &[String],
        config: &OptimizationConfig
    ) -> bool {
        // Check if current context is exempt
        if let Some(ctx) = context.last() {
            if config.custom_field_exempt_contexts.contains(ctx) {
                return false;
            }
        }

        // Check if field should be removed
        config.custom_fields_to_remove.contains(key) ||
            config.custom_field_patterns.iter().any(|pattern| {
                if pattern.ends_with('*') {
                    let prefix = &pattern[..pattern.len() - 1];
                    key.starts_with(prefix)
                } else {
                    key == pattern
                }
            })
    }

    fn process_custom_fields_static(
        value: Value,
        context: &[String],
        config: Arc<OptimizationConfig>,
        metrics: Arc<Mutex<&mut OptimizationMetrics>>,
    ) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                let items: Vec<_> = map.into_iter().collect();
                let mut new_map = Map::with_capacity(items.len());

                for (key, val) in items {
                    if !Self::should_remove_custom_field_static(&key, context, &config) {
                        let mut new_context = context.to_vec();
                        if val.is_object() || val.is_array() {
                            new_context.push(key.clone());
                        }
                        let processed_value = Self::process_custom_fields_static(
                            val,
                            &new_context,
                            config.clone(),
                            metrics.clone()
                        )?;
                        new_map.insert(key, processed_value);
                    } else {
                        if let Ok(mut metrics) = metrics.lock() {
                            *metrics.items_affected
                                .entry(OptimizationType::CustomFieldRemoval)
                                .or_insert(0) += 1;
                        }
                    }
                }
                Ok(Value::Object(new_map))
            }
            Value::Array(arr) => {
                let processed: Result<Vec<_>, _> = arr
                    .into_iter()
                    .map(|v| Self::process_custom_fields_static(
                        v,
                        context,
                        config.clone(),
                        metrics.clone()
                    ))
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

    fn compress_value(&mut self, value: &Value) -> Result<Vec<u8>, OptimizeError> {
        let json_str = serde_json::to_string(value)?;
        let _original_size = json_str.len();
        
        // First perform compression analysis
        let analysis = self.analyze_compression_potential(&json_str)?;
        
        // Check if compression is worthwhile
        if !self.should_compress(&analysis) {
            return Ok(json_str.into_bytes());
        }
    
        // Use the recommended compression level from analysis
        let mut encoder = flate2::write::ZlibEncoder::new(
            Vec::new(),
            flate2::Compression::new(analysis.recommended_level)
        );
        encoder.write_all(json_str.as_bytes())?;
        Ok(encoder.finish()?)
    }

    fn analyze_compression_potential(&self, data: &str) -> Result<CompressionAnalysis, OptimizeError> {
        let original_size = data.len();
        
        if original_size <= self.config.compression_threshold {
            return Ok(CompressionAnalysis {
                original_size,
                estimated_reduction_bps: 0,
                compression_ratio_bps: 10000, // 100% as basis points
                recommended_level: 0,
            });
        }
    
        // Calculate chunk size based on available threads
        let chunk_size = (original_size / rayon::current_num_threads()).max(1024);
        let chunks: Vec<_> = data.as_bytes()
            .chunks(chunk_size)
            .collect();
    
        // Process chunks in parallel
        let sample_results: Vec<Vec<(u32, u32)>> = chunks.par_iter()
            .map(|chunk| {
                // Test different compression levels
                (1..=9).into_par_iter()
                    .map(|level| {
                        let mut encoder = flate2::write::ZlibEncoder::new(
                            Vec::new(),
                            flate2::Compression::new(level)
                        );
                        encoder.write_all(chunk).unwrap();
                        let compressed = encoder.finish().unwrap();
                        // Convert ratio to basis points
                        let ratio_bps = ((compressed.len() * 10000) / chunk.len()) as u32;
                        (level, ratio_bps)
                    })
                    .collect()
            })
            .collect();
    
        // Calculate average ratios for each level
        let mut total_ratios = vec![0u64; 9];
        let mut count = 0u64;
        
        for result in &sample_results {
            for (idx, &(_, ratio)) in result.iter().enumerate() {
                total_ratios[idx] += ratio as u64;
            }
            count += 1;
        }
    
        let avg_ratios: Vec<u32> = total_ratios.iter()
            .map(|&total| (total / count) as u32)
            .collect();
    
        // Find best compression level
        let mut best_level = 6;
        let mut best_ratio_bps = avg_ratios[5];
        
        for (level, &ratio_bps) in avg_ratios.iter().enumerate().skip(6) {
            // Compare improvements in basis points
            if best_ratio_bps > 0 {  // Prevent division by zero
                let improvement_bps = ((best_ratio_bps - ratio_bps) * 10000) / best_ratio_bps;
                if improvement_bps > 500 { // 5% improvement threshold
                    best_level = level + 1;
                    best_ratio_bps = ratio_bps;
                }
            }
        }
    
        // Calculate final metrics
        let estimated_size = (original_size * best_ratio_bps as usize) / 10000;
        let reduction_bps = if original_size > 0 {
            ((original_size - estimated_size) * 10000) / original_size
        } else {
            0
        };
    
        Ok(CompressionAnalysis {
            original_size,
            estimated_reduction_bps: reduction_bps as u32,
            compression_ratio_bps: best_ratio_bps,
            recommended_level: best_level as u32,
        })
    }
    
    fn should_compress(&self, analysis: &CompressionAnalysis) -> bool {
        // Don't compress if below absolute size threshold
        if analysis.original_size <= self.config.compression_threshold {
            return false;
        }

        // Constants in basis points (1/100th of a percent)
        const MIN_REDUCTION_BPS: u32 = 200;   // 200 bytes = 2%
        const MIN_ABSOLUTE_SIZE: usize = 100;  // 100 bytes
        const MIN_REDUCTION_BYTES: usize = 30; // 30 bytes

        // Calculate absolute reduction in bytes
        let reduction_bytes = (analysis.original_size * analysis.estimated_reduction_bps as usize) / 10000;

        // Size and reduction criteria
        let should_compress = analysis.original_size >= MIN_ABSOLUTE_SIZE && 
            (analysis.estimated_reduction_bps >= MIN_REDUCTION_BPS || 
            reduction_bytes >= MIN_REDUCTION_BYTES);

        should_compress
    }

    fn compress_nested_parallel_static(
        value: &Value,
        context: &[String],
        config: Arc<OptimizationConfig>,
        metrics: Arc<Mutex<&mut OptimizationMetrics>>,
        original_size: usize,
    ) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                let items: Vec<_> = map.iter().collect();
                
                // Process nested items first
                let processed: Result<Vec<(String, Value)>, OptimizeError> = if items.len() > 20 {
                    items.par_iter()
                        .map(|(k, v)| -> Result<(String, Value), OptimizeError> {
                            let mut new_context = context.to_vec();
                            new_context.push(k.to_string());
                            
                            let processed_value = Self::compress_nested_parallel_static(
                                v,
                                &new_context,
                                config.clone(),
                                metrics.clone(),
                                original_size
                            )?;
                            
                            Ok((k.to_string(), processed_value))
                        })
                        .collect()
                } else {
                    items.iter()
                        .map(|(k, v)| -> Result<(String, Value), OptimizeError> {
                            let mut new_context = context.to_vec();
                            new_context.push(k.to_string());
                            
                            let processed_value = Self::compress_nested_parallel_static(
                                v,
                                &new_context,
                                config.clone(),
                                metrics.clone(),
                                original_size
                            )?;
                            
                            Ok((k.to_string(), processed_value))
                        })
                        .collect()
                };
    
                let mut new_map = Map::new();
                for (k, v) in processed? {
                    new_map.insert(k, v);
                }
    
                // Check if current object should be compressed
                let current_size = serde_json::to_string(&Value::Object(new_map.clone()))?.len();
                if current_size > config.compression_threshold {
                    let compressed = Self::compress_value_static(
                        &Value::Object(new_map),
                        config.compression_level
                    )?;
                    
                    // Update compression metrics
                    if let Ok(mut metrics) = metrics.lock() {
                        *metrics.items_affected
                            .entry(OptimizationType::Compression)
                            .or_insert(0) += 1;
                    }
    
                    let mut compressed_map = Map::new();
                    compressed_map.insert(
                        "compressed_data".to_string(),
                        Value::String(STANDARD.encode(&compressed))
                    );
                    compressed_map.insert(
                        "original_size".to_string(),
                        Value::Number(current_size.into())
                    );
                    Ok(Value::Object(compressed_map))
                } else {
                    Ok(Value::Object(new_map))
                }
            }
            Value::Array(arr) => {
                if arr.len() > 20 {
                    let processed: Result<Vec<_>, _> = arr
                        .par_iter()
                        .map(|v| Self::compress_nested_parallel_static(
                            v, 
                            context, 
                            config.clone(), 
                            metrics.clone(),
                            original_size
                        ))
                        .collect();
                    Ok(Value::Array(processed?))
                } else {
                    let processed: Result<Vec<_>, _> = arr
                        .iter()
                        .map(|v| Self::compress_nested_parallel_static(
                            v, 
                            context, 
                            config.clone(), 
                            metrics.clone(),
                            original_size
                        ))
                        .collect();
                    Ok(Value::Array(processed?))
                }
            }
            _ => Ok(value.clone()),
        }
    }
    
    fn compress_nested_parallel(
        &self,
        value: &Value,
        context: &[String],
        metrics: Arc<Mutex<&mut OptimizationMetrics>>,
    ) -> Result<Value, OptimizeError> {
        match value {
            Value::Object(map) => {
                if !self.config.compress_nested {
                    return Ok(value.clone());
                }
    
                let items: Vec<_> = map.iter().collect();
                
                if items.len() > 20 {
                    let processed: Result<Vec<(String, Value)>, OptimizeError> = items
                        .par_iter()
                        .map(|(k, v)| -> Result<(String, Value), OptimizeError> {
                            let mut new_context = context.to_vec();
                            new_context.push(k.to_string());
                            
                            let processed_value = self.compress_nested_parallel(
                                v,
                                &new_context,
                                metrics.clone()
                            )?;
                            
                            Ok((k.to_string(), processed_value))
                        })
                        .collect();
    
                    let mut new_map = Map::new();
                    for (k, v) in processed? {
                        new_map.insert(k, v);
                    }

                    let analysis = self.analyze_compression_potential(
                        &serde_json::to_string(&Value::Object(new_map.clone()))?
                    )?;
    
                    if self.should_compress(&analysis) {
                        let compressed = Self::compress_value_static(
                            &Value::Object(new_map),
                            self.config.compression_level,
                        )?;
                        
                        if let Ok(mut metrics) = metrics.lock() {
                            *metrics.items_affected
                                .entry(OptimizationType::Compression)
                                .or_insert(0) += 1;
                        }
    
                        let mut compressed_map = Map::new();
                        compressed_map.insert(
                            "compressed_data".to_string(),
                            Value::String(STANDARD.encode(&compressed))
                        );
                        compressed_map.insert(
                            "original_size".to_string(),
                            Value::Number(analysis.original_size.into())
                        );
                        Ok(Value::Object(compressed_map))
                    } else {
                        Ok(Value::Object(new_map))
                    }
                } else {
                    // Sequential processing remains the same
                    let mut new_map = Map::new();
                    for (k, v) in map {
                        let mut new_context = context.to_vec();
                        new_context.push(k.to_string());
                        let processed_value = self.compress_nested_parallel(
                            v,
                            &new_context,
                            metrics.clone()
                        )?;
                        new_map.insert(k.to_string(), processed_value);
                    }
                    Ok(Value::Object(new_map))
                }
            }
            Value::Array(arr) => {
                if arr.len() > 20 {
                    let processed: Result<Vec<_>, _> = arr
                        .par_iter()
                        .map(|v| self.compress_nested_parallel(v, context, metrics.clone()))
                        .collect();
                    Ok(Value::Array(processed?))
                } else {
                    let processed: Result<Vec<_>, _> = arr
                        .iter()
                        .map(|v| self.compress_nested_parallel(v, context, metrics.clone()))
                        .collect();
                    Ok(Value::Array(processed?))
                }
            }
            _ => Ok(value.clone()),
        }
    }
    
    fn compress_value_static(value: &Value, compression_level: u32) -> Result<Vec<u8>, OptimizeError> {
        let json_str = serde_json::to_string(value)?;
        let mut encoder = flate2::write::ZlibEncoder::new(
            Vec::new(),
            flate2::Compression::new(compression_level)
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
    ThreadError(String),
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