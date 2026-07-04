use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};
use std::collections::HashMap;
use xmltree::{Element, XMLNode};

fn strip_ns(tag: &str) -> String {
    tag.split('}').next_back().unwrap_or(tag).to_string()
}

fn node_text(element: &Element) -> Option<String> {
    let text = element
        .children
        .iter()
        .filter_map(|child| match child {
            XMLNode::Text(t) => Some(t.trim()),
            _ => None,
        })
        .collect::<Vec<&str>>()
        .join(" ");

    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn child_elements<'a>(element: &'a Element) -> Vec<&'a Element> {
    element
        .children
        .iter()
        .filter_map(|child| match child {
            XMLNode::Element(e) => Some(e),
            _ => None,
        })
        .collect()
}

fn flatten_element(element: &Element, parent_path: &str, include_attributes: bool) -> Map<String, Value> {
    let mut data = Map::new();
    let tag = strip_ns(&element.name);
    let path_prefix = if parent_path.is_empty() {
        tag
    } else {
        format!("{}.{}", parent_path, tag)
    };

    if include_attributes {
        for (k, v) in &element.attributes {
            data.insert(format!("{}.{}", path_prefix, strip_ns(k)), Value::String(v.clone()));
        }
    }

    let children = child_elements(element);
    if children.is_empty() {
        if let Some(text) = node_text(element) {
            data.insert(format!("{}.text", path_prefix), Value::String(text));
        }
        return data;
    }

    let mut grouped: HashMap<String, Vec<&Element>> = HashMap::new();
    for child in children {
        let key = strip_ns(&child.name);
        grouped.entry(key).or_default().push(child);
    }

    for (tag_name, siblings) in grouped {
        if siblings.len() == 1 {
            let child_data = flatten_element(siblings[0], &path_prefix, include_attributes);
            for (k, v) in child_data {
                data.insert(k, v);
            }
        } else {
            let items = siblings
                .into_iter()
                .map(|s| Value::Object(flatten_element(s, "", include_attributes)))
                .collect::<Vec<Value>>();
            data.insert(format!("{}.{}", path_prefix, tag_name), Value::Array(items));
        }
    }

    data
}

fn find_children_by_tag<'a>(elements: Vec<&'a Element>, tag: &str) -> Vec<&'a Element> {
    let mut next = Vec::new();
    for element in elements {
        for child in child_elements(element) {
            if strip_ns(&child.name) == tag {
                next.push(child);
            }
        }
    }
    next
}

fn collect_descendants_by_tag<'a>(element: &'a Element, tag: &str, output: &mut Vec<&'a Element>) {
    for child in child_elements(element) {
        if strip_ns(&child.name) == tag {
            output.push(child);
        }
        collect_descendants_by_tag(child, tag, output);
    }
}

fn parse_xml_records(
    xml_text: &str,
    record_path: Option<&str>,
    include_attributes: bool,
) -> Result<Vec<Map<String, Value>>, String> {
    let root = Element::parse(xml_text.as_bytes()).map_err(|e| format!("Invalid XML input: {e}"))?;

    if let Some(path) = record_path {
        let mut parts = path
            .trim_matches('.')
            .split('.')
            .filter(|p| !p.is_empty())
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        if parts.is_empty() {
            return Err("record_path cannot be empty".to_string());
        }

        let root_tag = strip_ns(&root.name);
        if parts.first().map(|p| p == &root_tag).unwrap_or(false) {
            parts.remove(0);
        }

        if parts.is_empty() {
            return Ok(vec![flatten_element(&root, "", include_attributes)]);
        }

        let parent_parts = &parts[..parts.len() - 1];
        let record_tag = &parts[parts.len() - 1];

        let mut parent_nodes = vec![&root];
        for parent_tag in parent_parts {
            parent_nodes = find_children_by_tag(parent_nodes, parent_tag);
        }

        if parent_nodes.is_empty() {
            return Err(format!(
                "Parent path '{}' not found in XML.",
                parent_parts.join(".")
            ));
        }

        let mut records = Vec::new();
        for parent in parent_nodes {
            let mut parent_data = Map::new();
            let parent_tag = strip_ns(&parent.name);

            if include_attributes {
                for (k, v) in &parent.attributes {
                    parent_data.insert(
                        format!("{}.{}", parent_tag, strip_ns(k)),
                        Value::String(v.clone()),
                    );
                }
            }

            if let Some(text) = node_text(parent) {
                parent_data.insert(format!("{}.text", parent_tag), Value::String(text));
            }

            let mut record_nodes = Vec::new();
            collect_descendants_by_tag(parent, record_tag, &mut record_nodes);

            for record in record_nodes {
                let record_data = flatten_element(record, "", include_attributes);
                let mut merged = parent_data.clone();
                for (k, v) in record_data {
                    merged.insert(k, v);
                }
                records.push(merged);
            }
        }

        Ok(records)
    } else {
        Ok(vec![flatten_element(&root, "", include_attributes)])
    }
}

#[pyfunction(signature = (xml_text, record_path=None, include_attributes=true))]
fn read_xml_records(
    xml_text: String,
    record_path: Option<String>,
    include_attributes: bool,
) -> PyResult<String> {
    match parse_xml_records(&xml_text, record_path.as_deref(), include_attributes) {
        Ok(records) => serde_json::to_string(&records)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize XML records: {e}"))),
        Err(message) => Err(PyValueError::new_err(message)),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_xml_records, m)?)?;
    Ok(())
}
