use std::io::{BufRead, ErrorKind, Lines};

use crate::graph::*;

pub type Result<T> = std::io::Result<T>;

impl Graph {
    pub fn try_read_pace<R: BufRead>(reader: R) -> Result<Self> {
        let pace_reader = PaceReader::try_new(reader)?;
        let n = pace_reader.number_of_nodes();
        Ok(Self::new(n, pace_reader))
    }
}

pub struct PaceReader<R> {
    lines: Lines<R>,
    number_of_nodes: Node,
    number_of_edges: usize,
}

#[allow(dead_code)]
impl<R: BufRead> PaceReader<R> {
    pub fn try_new(reader: R) -> Result<Self> {
        let mut pace_reader = Self {
            lines: reader.lines(),
            number_of_nodes: 0,
            number_of_edges: 0,
        };

        (pace_reader.number_of_nodes, pace_reader.number_of_edges) = pace_reader.parse_header()?;
        Ok(pace_reader)
    }

    pub fn number_of_nodes(&self) -> Node {
        self.number_of_nodes
    }
}

impl<R: BufRead> Iterator for PaceReader<R> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.parse_edge_line().unwrap().map(|(u, v)| (u - 1, v - 1))
    }
}

macro_rules! raise_error_unless {
    ($cond : expr, $kind : expr, $info : expr) => {
        if !($cond) {
            return Err(std::io::Error::new($kind, $info));
        }
    };
}

macro_rules! parse_next_value {
    ($iterator : expr, $name : expr) => {{
        let next = $iterator.next();
        raise_error_unless!(
            next.is_some(),
            ErrorKind::InvalidData,
            format!("Premature end of line when parsing {}.", $name)
        );

        let parsed = next.unwrap().parse();
        raise_error_unless!(
            parsed.is_ok(),
            ErrorKind::InvalidData,
            format!("Invalid value found. Cannot parse {}.", $name)
        );

        parsed.unwrap()
    }};
}

impl<R: BufRead> PaceReader<R> {
    fn next_non_comment_line(&mut self) -> Result<Option<String>> {
        loop {
            let line = self.lines.next();
            match line {
                None => return Ok(None),
                Some(Err(x)) => return Err(x),
                Some(Ok(line)) if line.starts_with('c') => continue,
                Some(Ok(line)) => return Ok(Some(line)),
            }
        }
    }

    fn parse_header(&mut self) -> Result<(Node, usize)> {
        let line = self.next_non_comment_line()?;

        raise_error_unless!(line.is_some(), ErrorKind::InvalidData, "No header found");
        let line = line.unwrap();

        let mut parts = line.split(' ').filter(|t| !t.is_empty());

        raise_error_unless!(
            parts.next().is_some_and(|t| t.starts_with('p')),
            ErrorKind::InvalidData,
            "Invalid header found; line should start with p"
        );

        raise_error_unless!(
            parts.next() == Some("ds"),
            ErrorKind::InvalidData,
            "Invalid header found; file type should be \"ds\""
        );

        let number_of_nodes = parse_next_value!(parts, "Header>Number of nodes");
        let number_of_edges = parse_next_value!(parts, "Header>Number of edges");

        raise_error_unless!(
            parts.next().is_none(),
            ErrorKind::InvalidData,
            "Invalid header found; expected end of line"
        );

        Ok((number_of_nodes, number_of_edges))
    }

    fn parse_edge_line(&mut self) -> Result<Option<Edge>> {
        let line = self.next_non_comment_line()?;
        if let Some(line) = line {
            let mut parts = line.split(' ').filter(|t| !t.is_empty());

            let from = parse_next_value!(parts, "Source node");
            let dest = parse_next_value!(parts, "Target node");

            debug_assert!((1..=self.number_of_nodes).contains(&from));
            debug_assert!((1..=self.number_of_nodes).contains(&dest));

            Ok(Some((from, dest)))
        } else {
            Ok(None)
        }
    }
}
