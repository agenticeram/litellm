use litellm_python_compat::lenient::{
    IsTrue, LaxBool, LaxBoolOr, Lenient, ParamNames, StrBool, TextEntries, Truthy,
};
use rstest::rstest;
use serde::Deserialize;
use serde_json::{Value, json};
use serde_with::serde_as;

#[serde_as]
#[derive(Deserialize)]
struct Flag(#[serde_as(as = "IsTrue")] bool);

/// Expected values from CPython: `value is True` for the JSON value loaded with `json.loads`.
#[rstest]
#[case::true_(json!(true), true)]
#[case::false_(json!(false), false)]
#[case::string_true(json!("true"), false)]
#[case::one(json!(1), false)]
#[case::null(json!(null), false)]
#[case::list_with_true(json!([true]), false)]
fn is_true_is_only_the_boolean_true(#[case] value: Value, #[case] expected: bool) {
    assert_eq!(serde_json::from_value::<Flag>(value).unwrap().0, expected);
}

#[serde_as]
#[derive(Deserialize)]
struct Bool(#[serde_as(as = "Truthy")] bool);

/// Expected values from CPython: `bool(value)` for the JSON value loaded with `json.loads`.
#[rstest]
#[case::true_(json!(true), true)]
#[case::false_(json!(false), false)]
#[case::zero(json!(0), false)]
#[case::one(json!(1), true)]
#[case::zero_float(json!(0.0), false)]
#[case::half(json!(0.5), true)]
#[case::empty_string(json!(""), false)]
#[case::string_false(json!("false"), true)]
#[case::null(json!(null), false)]
#[case::empty_list(json!([]), false)]
#[case::list_of_false(json!([false]), true)]
#[case::empty_mapping(json!({}), false)]
#[case::mapping(json!({"a": null}), true)]
fn truthy_is_python_bool(#[case] value: Value, #[case] expected: bool) {
    assert_eq!(serde_json::from_value::<Bool>(value).unwrap().0, expected);
}

#[rstest]
#[case::list(json!(["/a", "/b/*"]), Lenient::Valid(["/a".to_owned(), "/b/*".to_owned()].into()))]
#[case::empty(json!([]), Lenient::Valid([].into()))]
#[case::bare_string(json!("/a/*"), Lenient::Malformed)]
#[case::number(json!(5), Lenient::Malformed)]
#[case::mapping(json!({"a": 1}), Lenient::Malformed)]
#[case::non_string_entry(json!(["/a", 5, "/b/*"]), Lenient::Malformed)]
#[case::null_entry(json!(["/a", null]), Lenient::Malformed)]
fn a_value_of_another_shape_is_malformed(
    #[case] value: Value,
    #[case] expected: Lenient<Box<[String]>>,
) {
    assert_eq!(
        serde_json::from_value::<Lenient<Box<[String]>>>(value).unwrap(),
        expected
    );
}

#[rstest]
#[case::listed(Lenient::Valid(["/a".to_owned()].into()), &["/a"][..])]
#[case::malformed(Lenient::Malformed, &[][..])]
fn items_of_a_malformed_list_are_none(
    #[case] list: Lenient<Box<[String]>>,
    #[case] expected: &[&str],
) {
    assert_eq!(list.items(), expected);
}

#[serde_as]
#[derive(Deserialize)]
struct Enforced(#[serde_as(as = "LaxBoolOr<true>")] bool);

/// Expected values from pydantic 2.13.4: `TypeAdapter(bool).validate_python(value)`, or `True` where
/// it raises, for the JSON value loaded with `json.loads`.
#[rstest]
#[case::true_(json!(true), true)]
#[case::false_(json!(false), false)]
#[case::text_true(json!("true"), true)]
#[case::text_false(json!("false"), false)]
#[case::capitalized(json!("True"), true)]
#[case::padded_false(json!(" false"), true)]
#[case::no(json!("no"), false)]
#[case::off(json!("off"), false)]
#[case::text_zero(json!("0"), false)]
#[case::unknown_word(json!("maybe"), true)]
#[case::zero(json!(0), false)]
#[case::one(json!(1), true)]
#[case::two(json!(2), true)]
#[case::zero_float(json!(0.0), false)]
#[case::one_float(json!(1.0), true)]
#[case::null(json!(null), true)]
#[case::list(json!([]), true)]
#[case::mapping(json!({}), true)]
fn lax_bool_or_falls_back_where_pydantic_raises(#[case] value: Value, #[case] expected: bool) {
    assert_eq!(
        serde_json::from_value::<Enforced>(value).unwrap().0,
        expected
    );
}

#[serde_as]
#[derive(Deserialize)]
struct TextFlag(#[serde_as(as = "StrBool")] bool);

#[rstest]
#[case::flag(json!(true), true)]
#[case::false_flag(json!(false), false)]
#[case::true_text(json!(" True "), true)]
#[case::python_stripped_true_text(json!("\u{1c}true"), true)]
#[case::false_text(json!("false"), false)]
#[case::other_text(json!("yes"), false)]
#[case::number(json!(1), false)]
#[case::null(json!(null), false)]
fn str_bool_is_true_only_for_true(#[case] value: Value, #[case] expected: bool) {
    assert_eq!(
        serde_json::from_value::<TextFlag>(value).unwrap().0,
        expected
    );
}

#[serde_as]
#[derive(Deserialize)]
struct CoreLaxBool(#[serde_as(as = "LaxBool")] Option<bool>);

#[derive(Deserialize)]
struct FlagFixtures {
    rows: Vec<FlagRow>,
}

#[derive(Deserialize)]
struct FlagRow {
    input: String,
    lax_bool: Option<bool>,
}

/// Expected values from `generated/flags.json`, which `core_helpers.lax_bool` wrote.
#[test]
fn lax_bool_matches_core_helpers() {
    let fixtures: FlagFixtures = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/generated/flags.json"
    )))
    .unwrap();
    let mismatches: Vec<String> = fixtures
        .rows
        .iter()
        .filter_map(|row| {
            let read = serde_json::from_str::<CoreLaxBool>(&row.input).unwrap().0;
            (read != row.lax_bool).then(|| format!("{}: {read:?} != {:?}", row.input, row.lax_bool))
        })
        .collect();
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

#[serde_as]
#[derive(Deserialize)]
struct Names(#[serde_as(as = "ParamNames")] Box<[String]>);

/// Expected values from CPython: `allowed_openai_param_names(json.loads(value))`.
#[rstest]
#[case::text(json!("response_format, tools;x-y"), &["response_format", "tools", "x", "y"][..])]
#[case::mapping(json!({"response_format": null, "tools": 1}), &["response_format", "tools"][..])]
#[case::list(json!(["response_format", 5, null, "tools"]), &["response_format", "tools"][..])]
#[case::null(json!(null), &[][..])]
#[case::number(json!(5), &[][..])]
fn param_names_read_text_mapping_keys_and_list_entries(
    #[case] value: Value,
    #[case] expected: &[&str],
) {
    assert_eq!(
        &*serde_json::from_value::<Names>(value).unwrap().0,
        expected
    );
}

#[serde_as]
#[derive(Deserialize)]
struct Entries(#[serde_as(as = "TextEntries")] Box<[String]>);

#[rstest]
#[case::list(json!(["tools", 5, "response_format"]), &["tools", "response_format"][..])]
#[case::text(json!("tools"), &[][..])]
#[case::mapping(json!({"tools": 1}), &[][..])]
fn text_entries_are_a_lists_text_only(#[case] value: Value, #[case] expected: &[&str]) {
    assert_eq!(
        &*serde_json::from_value::<Entries>(value).unwrap().0,
        expected
    );
}
