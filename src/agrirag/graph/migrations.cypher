// Generated from agrirag.graph.schema - do not edit by hand.

// Crop
CREATE CONSTRAINT crop_id_unique IF NOT EXISTS FOR (n:Crop) REQUIRE n.id IS UNIQUE
;
// Region
CREATE CONSTRAINT region_id_unique IF NOT EXISTS FOR (n:Region) REQUIRE n.id IS UNIQUE
;
// ClimateZone
CREATE CONSTRAINT climatezone_id_unique IF NOT EXISTS FOR (n:ClimateZone) REQUIRE n.id IS UNIQUE
;
// SoilType
CREATE CONSTRAINT soiltype_id_unique IF NOT EXISTS FOR (n:SoilType) REQUIRE n.id IS UNIQUE
;
// Pest
CREATE CONSTRAINT pest_id_unique IF NOT EXISTS FOR (n:Pest) REQUIRE n.id IS UNIQUE
;
// Input
CREATE CONSTRAINT input_id_unique IF NOT EXISTS FOR (n:Input) REQUIRE n.id IS UNIQUE
;
// Practice
CREATE CONSTRAINT practice_id_unique IF NOT EXISTS FOR (n:Practice) REQUIRE n.id IS UNIQUE
;
// Regulation
CREATE CONSTRAINT regulation_id_unique IF NOT EXISTS FOR (n:Regulation) REQUIRE n.id IS UNIQUE
;

// Provenance nodes (populated in Phase 2)
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE
;
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE
;

// Fulltext index backing entity linking.
// Dropped first: CREATE ... IF NOT EXISTS silently keeps the old
// definition, so adding a label to FULLTEXT_LABELS would have no
// effect and the new label would stay permanently unlinkable.
DROP INDEX entity_names IF EXISTS
;
CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Crop|Region|ClimateZone|SoilType|Pest|Input|Practice) ON EACH [n.name, n.aliases]
;
