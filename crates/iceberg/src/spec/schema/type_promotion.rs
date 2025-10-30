// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Type promotion utilities for schema evolution.
//!
//! This module provides functions to check if a type can be promoted (widened)
//! to another type safely, following Iceberg's schema evolution rules.

use crate::spec::datatypes::{PrimitiveType, Type};

/// Check if a type can be promoted (widened) to another type.
///
/// Type promotion allows schema evolution while maintaining backward compatibility.
/// Only specific type promotions are allowed:
/// - int → long
/// - float → double
/// - decimal(P,S) → decimal(P',S) where P' > P (same scale, increased precision)
///
/// # Arguments
/// * `from_type` - The original type
/// * `to_type` - The target type to promote to
///
/// # Returns
/// `true` if the promotion is allowed, `false` otherwise
///
/// # Examples
/// ```
/// use iceberg::spec::datatypes::{PrimitiveType, Type};
/// use iceberg::spec::schema::is_promotion_allowed;
///
/// // int → long is allowed
/// assert!(is_promotion_allowed(
///     &Type::Primitive(PrimitiveType::Int),
///     &Type::Primitive(PrimitiveType::Long)
/// ));
///
/// // float → double is allowed
/// assert!(is_promotion_allowed(
///     &Type::Primitive(PrimitiveType::Float),
///     &Type::Primitive(PrimitiveType::Double)
/// ));
///
/// // long → int is NOT allowed (narrowing)
/// assert!(!is_promotion_allowed(
///     &Type::Primitive(PrimitiveType::Long),
///     &Type::Primitive(PrimitiveType::Int)
/// ));
/// ```
pub fn is_promotion_allowed(from_type: &Type, to_type: &Type) -> bool {
    // If types are equal, no promotion needed
    if from_type == to_type {
        return true;
    }

    // Only primitive types can be promoted
    let (Type::Primitive(from_prim), Type::Primitive(to_prim)) = (from_type, to_type) else {
        return false;
    };

    match (from_prim, to_prim) {
        // int → long
        (PrimitiveType::Int, PrimitiveType::Long) => true,

        // float → double
        (PrimitiveType::Float, PrimitiveType::Double) => true,

        // decimal(P,S) → decimal(P',S) where P' > P
        (
            PrimitiveType::Decimal {
                precision: p1,
                scale: s1,
            },
            PrimitiveType::Decimal {
                precision: p2,
                scale: s2,
            },
        ) => {
            // Scale must remain the same, precision can only increase
            s1 == s2 && p2 > p1
        }

        // All other combinations are not allowed
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::datatypes::{PrimitiveType, Type};

    #[test]
    fn test_same_type_is_allowed() {
        let int_type = Type::Primitive(PrimitiveType::Int);
        assert!(is_promotion_allowed(&int_type, &int_type));

        let string_type = Type::Primitive(PrimitiveType::String);
        assert!(is_promotion_allowed(&string_type, &string_type));
    }

    #[test]
    fn test_int_to_long() {
        let from = Type::Primitive(PrimitiveType::Int);
        let to = Type::Primitive(PrimitiveType::Long);
        assert!(is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_long_to_int_not_allowed() {
        let from = Type::Primitive(PrimitiveType::Long);
        let to = Type::Primitive(PrimitiveType::Int);
        assert!(!is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_float_to_double() {
        let from = Type::Primitive(PrimitiveType::Float);
        let to = Type::Primitive(PrimitiveType::Double);
        assert!(is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_double_to_float_not_allowed() {
        let from = Type::Primitive(PrimitiveType::Double);
        let to = Type::Primitive(PrimitiveType::Float);
        assert!(!is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_decimal_precision_promotion() {
        let from = Type::Primitive(PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        });
        let to = Type::Primitive(PrimitiveType::Decimal {
            precision: 20,
            scale: 2,
        });
        assert!(is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_decimal_scale_change_not_allowed() {
        let from = Type::Primitive(PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        });
        let to = Type::Primitive(PrimitiveType::Decimal {
            precision: 10,
            scale: 3,
        });
        assert!(!is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_decimal_precision_decrease_not_allowed() {
        let from = Type::Primitive(PrimitiveType::Decimal {
            precision: 20,
            scale: 2,
        });
        let to = Type::Primitive(PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        });
        assert!(!is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_int_to_float_not_allowed() {
        let from = Type::Primitive(PrimitiveType::Int);
        let to = Type::Primitive(PrimitiveType::Float);
        assert!(!is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_string_to_int_not_allowed() {
        let from = Type::Primitive(PrimitiveType::String);
        let to = Type::Primitive(PrimitiveType::Int);
        assert!(!is_promotion_allowed(&from, &to));
    }

    #[test]
    fn test_non_primitive_types_not_allowed() {
        use crate::spec::datatypes::{ListType, NestedField, StructType};

        // Different struct types should not be allowed
        let from = Type::Struct(StructType::new(vec![
            NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
        ]));
        let to = Type::Struct(StructType::new(vec![
            NestedField::required(2, "b", Type::Primitive(PrimitiveType::String)).into(),
        ]));
        assert!(!is_promotion_allowed(&from, &to));

        // Even same list types can't be "promoted" (they can only be equal)
        let list_type1 = Type::List(ListType {
            element_field: NestedField::list_element(1, Type::Primitive(PrimitiveType::Int), false)
                .into(),
        });
        let list_type2 = Type::List(ListType {
            element_field: NestedField::list_element(
                2,
                Type::Primitive(PrimitiveType::Long),
                false,
            )
            .into(),
        });
        assert!(!is_promotion_allowed(&list_type1, &list_type2));
    }
}
